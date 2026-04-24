import json
import pickle
import struct
from typing import Any, Dict, List, Tuple
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
import time


import pyarrow.dataset as ds  # type: ignore



def _is_null(v: Any) -> bool:
    if v is None:
        return True
    try:
        return isinstance(v, float) and np.isnan(v)
    except Exception:
        return False


def load_feat_dict_from_parquet_folder(
    src: Path, id_col: str, feat_ids: List[str], *, keep_all_rows: bool = True
) -> Dict[str, Dict[str, Any]]:
    """
    从 parquet(folder/file) 读取特征表，返回 dict: {str(id): {feat_id: feat_value},...}。
    - 空值不会写入 feature dict（交由 fill_missing_feat 补默认值）
    - keep_all_rows=True 时，即使该行所有 feature 都为空，也会保留空 dict，避免后续采样/查找失败
    """
    if ds is None:
        raise RuntimeError("pyarrow 未安装，无法读取 parquet 数据。")
    dataset = ds.dataset(str(src), format="parquet")
    cols = [id_col] + feat_ids
    table = dataset.to_table(columns=cols)
    data = table.to_pydict()
    ids = data[id_col]
    out: Dict[str, Dict[str, Any]] = {}
    for i in tqdm(range(len(ids))):
        raw_id = ids[i]
        if _is_null(raw_id):
            continue
        key = str(raw_id)
        feat: Dict[str, Any] = {}
        for fid in feat_ids:
            v = data[fid][i]
            if _is_null(v):
                continue
            feat[fid] = v
        if feat or keep_all_rows:
            out[key] = feat
    return out


def load_mm_emb_from_parquet_folder(src: Path) -> Dict[str, np.ndarray]:
    """
    从 parquet(folder/file) 读取多模态 embedding：
    列为 anonymous_cid, emb；emb 为向量(list/np.ndarray)。
    """
    if ds is None:
        raise RuntimeError("pyarrow 未安装，无法读取 parquet 数据。")
    dataset = ds.dataset(str(src), format="parquet")
    table = dataset.to_table(columns=["anonymous_cid", "emb"])
    data = table.to_pydict()
    cids = data["anonymous_cid"]
    embs = data["emb"]
    out: Dict[str, np.ndarray] = {}
    for cid, emb in zip(cids, embs):
        if _is_null(cid) or _is_null(emb):
            continue
        if isinstance(emb, np.ndarray):
            out[str(cid)] = emb.astype(np.float32)
        elif isinstance(emb, str):
            out[str(cid)] = np.asarray(json.loads(emb), dtype=np.float32)
        else:
            out[str(cid)] = np.asarray(emb, dtype=np.float32)
    return out


class MyDataset(torch.utils.data.Dataset):
    """
    用户序列数据集

    Args:
        data_dir: 数据文件目录
        args: 全局参数

    Attributes:
        data_dir: 数据文件目录
        maxlen: 最大长度
        item_feat_dict: 物品特征字典
        mm_emb_ids: 激活的mm_emb特征ID
        mm_emb_dict: 多模态特征字典
        itemnum: 物品数量
        usernum: 用户数量
        indexer_i_rev: 物品索引字典 (reid -> item_id)
        indexer_u_rev: 用户索引字典 (reid -> user_id)
        indexer: 索引字典
        feature_default_value: 特征缺省值
        feature_types: 特征类型，分为user和item的sparse, array, emb, continual类型
        feat_statistics: 特征统计信息，包括user和item的特征数量
    """

    def __init__(self, data_dir, args):
        """
        初始化数据集
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        # 对于 parquet seq：测试集需要保留原始 user_id(str) 以便输出；训练集默认不需要
        self.keep_raw_user_id = False
        self.full_events, self.user_indices = load_seq_as_list(self.data_dir / "seq")
        self.seq_usernum = len(self.user_indices)
        self.maxlen = args.maxlen
        self.mm_emb_ids = args.mm_emb_id

        # 新数据：item/user 特征与 mm_emb 都来自 parquet(folder/file)
        # item/user feature_id 集合固定，避免在 indexer 初始化前访问 _init_feat_info
        self._item_sparse_ids = [
            '100',
            '117',
            # '111',
            '118',
            '101',
            '102',
            '119',
            '120',
            '114',
            '112',
            '121',
            '115',
            '122',
            '116',
        ]
        self._user_sparse_ids = ['103', '104', '105', '109']
        self._user_array_ids = ['106', '107', '108', '110']

        # item_feat_src = _find_first_existing(self.data_dir, ["item_feat"])
        item_feat_src = Path(self.data_dir, "item_feat")
        self.item_feat_dict = load_feat_dict_from_parquet_folder(
            item_feat_src, "item_id", self._item_sparse_ids, keep_all_rows=True
        )

        user_feat_src = Path(self.data_dir, "user_feat")
        self.user_feat_dict: Dict[str, Dict[str, Any]] = {}
        self.user_feat_dict = load_feat_dict_from_parquet_folder(
            user_feat_src,
            "user_id",
            self._user_sparse_ids + self._user_array_ids,
            keep_all_rows=True,
        )

        self.mm_emb_dict = load_mm_emb_v2(self.data_dir, self.mm_emb_ids)
        with open(self.data_dir / 'indexer.pkl', 'rb') as ff:
            indexer = pickle.load(ff)
            self.itemnum = len(indexer['i'])
            print(f"self.itemnum={self.itemnum}")
            self.usernum = len(indexer['u'])
            print(f"self.usernum={self.usernum}")
        self.indexer_i_rev = {v: k for k, v in indexer['i'].items()}
        self.indexer_u_rev = {v: k for k, v in indexer['u'].items()}
        self.indexer = indexer

        self.feature_default_value, self.feature_types, self.feat_statistics = self._init_feat_info()

    def new_load_user_data(self, uid):
        """
        注意 uid 是 reid
        """
        # start_time = time.time()
        if uid not in self.user_indices:
            return None

        start_idx, length = self.user_indices[uid]
        # [item_id, action_type, timestamp],... 
        user_data = self.full_events[start_idx:start_idx + length]

        user_feat = self.user_feat_dict.get(str(uid), {})

        data = []
        last_timestamp = None

        for row in user_data:
            item_id = int(row['item_id'])
            action_type = int(row['action_type'])
            timestamp = int(row['timestamp'])

            item_feat = self.item_feat_dict.get(str(item_id), {})
            data.append((uid, item_id, None, item_feat, action_type, timestamp))
            last_timestamp = timestamp
        data.append((uid, None, user_feat, None, None, last_timestamp))
        
        # end_time = time.time()
        # print(f"load user data time: {end_time - start_time}")
        return data

    def _load_user_data(self, uid):
        """
        从数据文件中加载单个用户的数据

        Args:
            uid: 用户ID(reid)

        Returns:
            data: 用户序列数据，格式为[(user_id, item_id, user_feat, item_feat, action_type, timestamp)]
        """
        start_time = time.time()
        row = self.seq_reader.get(uid)
        user_id = row.user_id

        # user feature：优先用 raw key，否则尝试 reid key
        user_feat = self.user_feat_dict.get(str(user_id), {})
        seq = row.seq or []


        data: List[Tuple[Any, Any, Any, Any, Any, Any]] = []
        for e in seq:
            if e is None:
                continue
            if isinstance(e, dict):
                item_id = e.get("item_id", 0)
                action_type = e.get("action_type", None)
                timestamp = e.get("timestamp", None)
            else:
                try:
                    item_id, action_type, timestamp = e
                except Exception:
                    raise ValueError(f"Invalid item data: {e}")

            item_feat = self.item_feat_dict.get(str(item_id), {})

            data.append((user_id, item_id, None, item_feat, action_type, timestamp))
        data.append((user_id, None, user_feat, None, None, timestamp))
        
        end_time = time.time()
        return data

    def _random_neq(self, l, r, s):
        """
        生成一个不在序列s中的随机整数, 用于训练时的负采样

        Args:
            l: 随机整数的最小值
            r: 随机整数的最大值
            s: 序列

        Returns:
            t: 不在序列s中的随机整数
        """
        
        t = np.random.randint(l, r)
        
        # cnt = 0
        while t in s or str(t) not in self.item_feat_dict:
            t = np.random.randint(l, r)
        return t

    def __getitem__(self, uid):
        """
        获取单个用户的数据，并进行padding处理，生成模型需要的数据格式

        Args:
            uid: 用户ID(reid)

        Returns:
            seq: 用户序列ID
            pos: 正样本ID（即下一个真实访问的item）
            neg: 负样本ID
            token_type: 用户序列类型，1表示item，2表示user
            next_token_type: 下一个token类型，1表示item，2表示user
            seq_feat: 用户序列特征，每个元素为字典，key为特征ID，value为特征值
            pos_feat: 正样本特征，每个元素为字典，key为特征ID，value为特征值
            neg_feat: 负样本特征，每个元素为字典，key为特征ID，value为特征值
        """
        uid += 1
        user_sequence = self.new_load_user_data(uid)  # 动态加载用户数据


        ext_user_sequence = []
        for record_tuple in user_sequence:
            u, i, user_feat, item_feat, action_type, _ = record_tuple
            if u and user_feat:
                ext_user_sequence.insert(0, (u, user_feat, 2, action_type))
            if i and item_feat:
                ext_user_sequence.append((i, item_feat, 1, action_type))

        seq = np.zeros([self.maxlen + 1], dtype=np.int32)
        pos = np.zeros([self.maxlen + 1], dtype=np.int32)
        neg = np.zeros([self.maxlen + 1], dtype=np.int32)
        token_type = np.zeros([self.maxlen + 1], dtype=np.int32)
        next_token_type = np.zeros([self.maxlen + 1], dtype=np.int32)
        next_action_type = np.zeros([self.maxlen + 1], dtype=np.int32)

        seq_feat = np.empty([self.maxlen + 1], dtype=object)
        pos_feat = np.empty([self.maxlen + 1], dtype=object)
        neg_feat = np.empty([self.maxlen + 1], dtype=object)

        nxt = ext_user_sequence[-1]
        idx = self.maxlen

        ts = set()
       
        for record_tuple in ext_user_sequence:
            if record_tuple[2] == 1 and record_tuple[0]:
                ts.add(record_tuple[0])

        # left-padding, 从后往前遍历，将用户序列填充到maxlen+1的长度
        for record_tuple in reversed(ext_user_sequence[:-1]):
            i, feat, type_, act_type = record_tuple
            next_i, next_feat, next_type, next_act_type = nxt
            feat = self.fill_missing_feat(feat, i)
            next_feat = self.fill_missing_feat(next_feat, next_i)
            seq[idx] = i
            token_type[idx] = type_
            next_token_type[idx] = next_type
            if next_act_type is not None:
                next_action_type[idx] = next_act_type
            seq_feat[idx] = feat
            if next_type == 1 and next_i != 0:
                pos[idx] = next_i
                pos_feat[idx] = next_feat
                neg_id = self._random_neq(1, self.itemnum + 1, ts)
                neg[idx] = neg_id
                neg_feat[idx] = self.fill_missing_feat(self.item_feat_dict[str(neg_id)], neg_id)
            nxt = record_tuple
            idx -= 1
            if idx == -1:
                break

        seq_feat = np.where(seq_feat == None, self.feature_default_value, seq_feat)
        pos_feat = np.where(pos_feat == None, self.feature_default_value, pos_feat)
        neg_feat = np.where(neg_feat == None, self.feature_default_value, neg_feat)

        return seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat

    def __len__(self):
        """
        返回数据集长度，即用户数量

        Returns:
            usernum: 用户数量
        """
        return self.seq_usernum

    def _init_feat_info(self):
        """
        初始化特征信息, 包括特征缺省值和特征类型

        Returns:
            feat_default_value: 特征缺省值，每个元素为字典，key为特征ID，value为特征缺省值
            feat_types: 特征类型，key为特征类型名称，value为包含的特征ID列表
        """
        feat_default_value = {}
        feat_statistics = {}
        feat_types = {}
        feat_types['user_sparse'] = ['103', '104', '105', '109']
        feat_types['item_sparse'] = [
            '100',
            '117',
            # '111',
            '118',
            '101',
            '102',
            '119',
            '120',
            '114',
            '112',
            '121',
            '115',
            '122',
            '116',
        ]
        feat_types['item_array'] = []
        feat_types['user_array'] = ['106', '107', '108', '110']
        feat_types['item_emb'] = self.mm_emb_ids
        feat_types['user_continual'] = []
        feat_types['item_continual'] = []

        for feat_id in feat_types['user_sparse']:
            feat_default_value[feat_id] = 0
            feat_statistics[feat_id] = len(self.indexer['f'][feat_id])
        for feat_id in feat_types['item_sparse']:
            feat_default_value[feat_id] = 0
            feat_statistics[feat_id] = len(self.indexer['f'][feat_id])
        for feat_id in feat_types['item_array']:
            feat_default_value[feat_id] = [0]
            feat_statistics[feat_id] = len(self.indexer['f'][feat_id])
        for feat_id in feat_types['user_array']:
            feat_default_value[feat_id] = [0]
            feat_statistics[feat_id] = len(self.indexer['f'][feat_id])
        for feat_id in feat_types['user_continual']:
            feat_default_value[feat_id] = 0
        for feat_id in feat_types['item_continual']:
            feat_default_value[feat_id] = 0
        for feat_id in feat_types['item_emb']:
            feat_default_value[feat_id] = np.zeros(
                list(self.mm_emb_dict[feat_id].values())[0].shape[0], dtype=np.float32
            )

        return feat_default_value, feat_types, feat_statistics

    def fill_missing_feat(self, feat, item_id):
        """
        对于原始数据中缺失的特征进行填充缺省值

        Args:
            feat: 特征字典
            item_id: 物品ID

        Returns:
            filled_feat: 填充后的特征字典
        """
        if feat == None:
            feat = {}
        filled_feat = {}
        for k in feat.keys():
            filled_feat[k] = feat[k]

        all_feat_ids = []
        for feat_type in self.feature_types.values():
            all_feat_ids.extend(feat_type)
        missing_fields = set(all_feat_ids) - set(feat.keys())
        for feat_id in missing_fields:
            filled_feat[feat_id] = self.feature_default_value[feat_id]
        for feat_id in self.feature_types['item_emb']:
            if item_id != 0 and self.indexer_i_rev[item_id] in self.mm_emb_dict[feat_id]:
                if type(self.mm_emb_dict[feat_id][self.indexer_i_rev[item_id]]) == np.ndarray:
                    filled_feat[feat_id] = self.mm_emb_dict[feat_id][self.indexer_i_rev[item_id]]

        return filled_feat

    @staticmethod
    def collate_fn(batch):
        """
        Args:
            batch: 多个__getitem__返回的数据

        Returns:
            seq: 用户序列ID, torch.Tensor形式
            pos: 正样本ID, torch.Tensor形式
            neg: 负样本ID, torch.Tensor形式
            token_type: 用户序列类型, torch.Tensor形式
            next_token_type: 下一个token类型, torch.Tensor形式
            seq_feat: 用户序列特征, list形式
            pos_feat: 正样本特征, list形式
            neg_feat: 负样本特征, list形式
        """
        seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat = zip(*batch)
        seq = torch.from_numpy(np.array(seq))
        pos = torch.from_numpy(np.array(pos))
        neg = torch.from_numpy(np.array(neg))
        token_type = torch.from_numpy(np.array(token_type))
        next_token_type = torch.from_numpy(np.array(next_token_type))
        next_action_type = torch.from_numpy(np.array(next_action_type))
        seq_feat = list(seq_feat)
        pos_feat = list(pos_feat)
        neg_feat = list(neg_feat)
        return seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat


class MyTestDataset(MyDataset):
    """
    测试数据集
    """

    def __init__(self, data_dir, args):
        super().__init__(data_dir, args)
        # 推理阶段需要保留 user_id(str) 以便输出对照
        self.keep_raw_user_id = True

    def _process_cold_start_feat(self, feat):
        """
        处理冷启动特征。训练集未出现过的特征value为字符串，默认转换为0.可设计替换为更好的方法。
        """
        processed_feat = {}
        for feat_id, feat_value in feat.items():
            if type(feat_value) == list:
                value_list = []
                for v in feat_value:
                    if type(v) == str:
                        value_list.append(0)
                    else:
                        value_list.append(v)
                processed_feat[feat_id] = value_list
            elif type(feat_value) == str:
                processed_feat[feat_id] = 0
            else:
                processed_feat[feat_id] = feat_value
        return processed_feat

    def __getitem__(self, uid):
        """
        获取单个用户的数据，并进行padding处理，生成模型需要的数据格式

        Args:
            uid: 用户在self.data_file中储存的行号
        Returns:
            seq: 用户序列ID
            token_type: 用户序列类型，1表示item，2表示user
            seq_feat: 用户序列特征，每个元素为字典，key为特征ID，value为特征值
            user_id: user_id eg. user_xxxxxx ,便于后面对照答案
        """
        # start_time = time.time()
        # user_sequence = self._load_user_data(uid) 
        uid += 1 # 会是从0开始的，但是数据从1开始
        #  # 动态加载用户数据
        user_sequence = self.new_load_user_data(uid) 

        user_id = "unknown"
        ext_user_sequence = []
        for record_tuple in user_sequence:
            u, i, user_feat, item_feat, _, _ = record_tuple
            if u:
                if type(u) == str:  # 如果是字符串，说明是user_id
                    user_id = u
                else:  # 如果是int，说明是re_id
                    user_id = self.indexer_u_rev[u]
            if u and user_feat:
                if type(u) == str:
                    u = 0
                if user_feat:
                    user_feat = self._process_cold_start_feat(user_feat)
                ext_user_sequence.insert(0, (u, user_feat, 2))

            if i and item_feat:
                # 序列对于训练时没见过的item，不会直接赋0，而是保留creative_id，creative_id远大于训练时的itemnum
                if i > self.itemnum:
                    i = 0
                if item_feat:
                    item_feat = self._process_cold_start_feat(item_feat)
                ext_user_sequence.append((i, item_feat, 1))

        seq = np.zeros([self.maxlen + 1], dtype=np.int32)
        token_type = np.zeros([self.maxlen + 1], dtype=np.int32)
        seq_feat = np.empty([self.maxlen + 1], dtype=object)

        idx = self.maxlen

        ts = set()
        for record_tuple in ext_user_sequence:
            if record_tuple[2] == 1 and record_tuple[0]:
                ts.add(record_tuple[0])

        for record_tuple in reversed(ext_user_sequence[:-1]):
            i, feat, type_ = record_tuple
            feat = self.fill_missing_feat(feat, i)
            seq[idx] = i
            token_type[idx] = type_
            seq_feat[idx] = feat
            idx -= 1
            if idx == -1:
                break

        seq_feat = np.where(seq_feat == None, self.feature_default_value, seq_feat)
        # end_time = time.time()
        # print(f"Time taken to process user {user_id}: {end_time - start_time:.2f}s")
        return seq, token_type, seq_feat, user_id

    def __len__(self):
        """
        Returns:
            len(self.seq_offsets): 用户数量
        """
        # return self.seq_reader.num_rows
        return self.seq_usernum

    @staticmethod
    def collate_fn(batch):
        """
        将多个__getitem__返回的数据拼接成一个batch

        Args:
            batch: 多个__getitem__返回的数据

        Returns:
            seq: 用户序列ID, torch.Tensor形式
            token_type: 用户序列类型, torch.Tensor形式
            seq_feat: 用户序列特征, list形式
            user_id: user_id, str
        """
        seq, token_type, seq_feat, user_id = zip(*batch)
        seq = torch.from_numpy(np.array(seq))
        token_type = torch.from_numpy(np.array(token_type))
        seq_feat = list(seq_feat)

        return seq, token_type, seq_feat, user_id


def save_emb(emb, save_path):
    """
    将Embedding保存为二进制文件

    Args:
        emb: 要保存的Embedding，形状为 [num_points, num_dimensions]
        save_path: 保存路径
    """
    num_points = emb.shape[0]  # 数据点数量
    num_dimensions = emb.shape[1]  # 向量的维度
    print(f'saving {save_path}')
    with open(Path(save_path), 'wb') as f:
        f.write(struct.pack('II', num_points, num_dimensions))
        emb.tofile(f)


def load_mm_emb(mm_path, feat_ids):
    """
    加载多模态特征Embedding

    Args:
        mm_path: 多模态特征Embedding路径
        feat_ids: 要加载的多模态特征ID列表

    Returns:
        mm_emb_dict: 多模态特征Embedding字典，key为特征ID，value为特征Embedding字典（key为item ID，value为Embedding）
    """
    SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}
    mm_emb_dict = {}
    for feat_id in tqdm(feat_ids, desc='Loading mm_emb'):
        shape = SHAPE_DICT[feat_id]
        emb_dict = {}
        if feat_id != '81':
            try:
                base_path = Path(mm_path, f'emb_{feat_id}_{shape}')
                for json_file in base_path.glob('*.json'):
                    with open(json_file, 'r', encoding='utf-8') as file:
                        for line in file:
                            data_dict_origin = json.loads(line.strip())
                            insert_emb = data_dict_origin['emb']
                            if isinstance(insert_emb, list):
                                insert_emb = np.array(insert_emb, dtype=np.float32)
                            data_dict = {data_dict_origin['anonymous_cid']: insert_emb}
                            emb_dict.update(data_dict)
            except Exception as e:
                print(f"transfer error: {e}")
        if feat_id == '81':
            with open(Path(mm_path, f'emb_{feat_id}_{shape}.pkl'), 'rb') as f:
                emb_dict = pickle.load(f)
        mm_emb_dict[feat_id] = emb_dict
        print(f'Loaded #{feat_id} mm_emb')
    return mm_emb_dict


def load_mm_emb_v2(data_dir: Path, feat_ids: List[str]) -> Dict[str, Dict[str, np.ndarray]]:
    """
    新格式：从 data_dir 下的 mm_emb_xx 目录读取（每个 feat_id 一个目录），目录内为多个 parquet part。
    旧格式：回退到原 load_mm_emb(creative_emb)。
    """
    SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}
    mm_emb_dict: Dict[str, Dict[str, np.ndarray]] = {}
    for feat_id in tqdm(feat_ids, desc="Loading mm_emb"):
        shape = SHAPE_DICT[feat_id]
        cand = Path(data_dir, "mm_emb", f"emb_{feat_id}_{shape}_parquet") 
        mm_emb_dict[feat_id] = load_mm_emb_from_parquet_folder(cand)
        print(f"Loaded #{feat_id} mm_emb from parquet: {cand}")
    return mm_emb_dict

def load_seq_in_batches(seq_dir, batch_size=100000):
    """分批加载序列数据"""
    dataset = ds.dataset(seq_dir, format="parquet")
    
    seq_user_ids = []
    seq_data_list = []
    
    # 使用 scanner 分批读取
    scanner = dataset.scanner(columns=["user_id", "seq"], batch_size=batch_size)

    for i, batch in enumerate(tqdm(scanner.to_batches(), desc="Loading seq batches", total=(dataset.count_rows() + batch_size - 1) // batch_size)):
        batch_dict = batch.to_pydict()
        seq_user_ids.extend(batch_dict["user_id"])
        seq_data_list.extend(batch_dict["seq"])
        
        # 可以在这里处理每个 batch，避免内存爆炸
        if i % 10 == 0:
            print(f"Loaded {len(seq_user_ids)} records, memory usage: {len(seq_user_ids) * 8 / 1e9:.2f} GB")
    
    return seq_user_ids, seq_data_list

def load_seq_as_list(seq_dir, batch_size=100000):
    """
    分批加载序列数据并构建为数组
    
    Returns:
        all_events: 所有事件的数组
        user_indices: {user_id: (start_idx, length)} 的字典
    """
    dataset = ds.dataset(seq_dir, format="parquet")
    scanner = dataset.scanner(columns=["user_id", "seq"], batch_size=batch_size)

    event_type = np.dtype([
        ('item_id', np.int32),
        ('action_type', np.int8),
        ('timestamp', np.int64)
    ])

    total_rows = dataset.count_rows()
    print(f"all the rows: {total_rows}")

    all_events_list = []
    user_indices = {} # {user_id: (start_idx, length)}

    cur_global_idx = 0

    with tqdm(total=total_rows, desc="Loading Seqs", unit=" rows") as pbar:

        for batch in scanner.to_batches():
            uids = batch.column("user_id").to_numpy()
            # user_ids = batch.column("user_id")
            seqs = batch.column("seq")

            flat_structs = seqs.values

            batch_events = np.empty(len(flat_structs), dtype=event_type)
            
            batch_events['item_id'] = flat_structs.field('item_id').to_numpy()
            batch_events['action_type'] = flat_structs.field('action_type').to_numpy()
            batch_events['timestamp'] = flat_structs.field('timestamp').to_numpy()

            all_events_list.append(batch_events)

            # 每个用户的序列偏移量
            offsets = seqs.offsets.to_numpy()
            lengths = np.diff(offsets)

            for i in range(len(uids)):
                uid = uids[i]
                if uid is not None:
                    user_indices[uid] = (cur_global_idx + offsets[i], lengths[i])
            
            cur_global_idx += len(batch_events)
            pbar.update(batch.num_rows)
    
    # 合并所有批次的event data
    full_events = np.concatenate(all_events_list)

    print(f"Memory Usage: {full_events.nbytes / 1024**2:.2f} MB (Events Only)")
    print(f"Total users: {len(user_indices)}")
    
    return full_events, user_indices

