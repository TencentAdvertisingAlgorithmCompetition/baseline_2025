import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import MyDataset
from model import BaselineModel

# -------------------- New: checkpoint helpers --------------------

def _ckpt_dir() -> Path:
    return Path(os.environ.get("TRAIN_CKPT_PATH", "./checkpoints/"))


def _list_ckpts() -> list[Path]:
    d = _ckpt_dir()
    if not d.exists():
        return []
    # files like: ckpt_epoch0003_step00001234.pt
    return sorted(d.glob("ckpt_epoch*_step*.pt"))


def _save_ckpt(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    args: argparse.Namespace,
    extra: dict | None = None,
) -> Path:
    d = _ckpt_dir()
    d.mkdir(parents=True, exist_ok=True)
    # ckpt_path = d / f"ckpt_epoch{epoch:04d}_step{global_step:08d}.pt"

    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "args": vars(args),
        "time": time.time(),
    }
    if extra:
        payload.update(extra)

    # torch.save(payload, ckpt_path)

    # best-effort "latest" pointer
    latest_path = d / "latest.pt"
    torch.save(payload, latest_path)
    


def _load_ckpt(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    ckpt_path: Path,
) -> tuple[int, int]:
    ckpt = torch.load(ckpt_path, map_location=torch.device(device))
    model.load_state_dict(ckpt["model"], strict=True)
    optimizer.load_state_dict(ckpt["optimizer"])

    # move optimizer state tensors to correct device
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)

    epoch = int(ckpt.get("epoch", 0))
    global_step = int(ckpt.get("global_step", 0))
    return epoch, global_step



def get_args():
    parser = argparse.ArgumentParser()

    # Train params
    parser.add_argument('--batch_size', default=2048, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--maxlen', default=101, type=int)

    # Baseline Model construction
    parser.add_argument('--hidden_units', default=32, type=int)
    parser.add_argument('--num_blocks', default=1, type=int)
    parser.add_argument('--num_epochs', default=3, type=int)
    parser.add_argument('--num_heads', default=1, type=int)
    parser.add_argument('--dropout_rate', default=0.2, type=float)
    parser.add_argument('--l2_emb', default=0.0, type=float)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--inference_only', action='store_true')
    parser.add_argument('--state_dict_path', default=None, type=str)
    parser.add_argument('--norm_first', action='store_true')

    # MMemb Feature ID
    parser.add_argument('--mm_emb_id', nargs='+', default=['81'], type=str, choices=[str(s) for s in range(81, 87)])

    # New: checkpointing/resume
    parser.add_argument('--save_every_steps', default=500, type=int)
    parser.add_argument('--resume', action='store_true', help='resume from latest checkpoint in TRAIN_CKPT_PATH')
    parser.add_argument('--resume_path', default=None, type=str, help='explicit checkpoint path (.pt)')


    args = parser.parse_args()

    return args


if __name__ == '__main__':
    run_id = time.strftime("%Y%m%d_%H%M%S")
    # run_id = "20260129_102310"
    os.environ["TRAIN_LOG_PATH"] = f"./logs/{run_id}"
    os.environ["TRAIN_TF_EVENTS_PATH"] = f"./tf_events/{run_id}"
    os.environ["TRAIN_CKPT_PATH"] = f"./checkpoints/{run_id}"
    # global dataset
    os.environ["TRAIN_DATA_PATH"] = "/apdcephfs_szgm/share_303492287/ryanylsun/TencentGR/second/second"

    Path(os.environ.get('TRAIN_LOG_PATH')).mkdir(parents=True, exist_ok=True)
    Path(os.environ.get('TRAIN_TF_EVENTS_PATH')).mkdir(parents=True, exist_ok=True)
    Path(os.environ.get('TRAIN_CKPT_PATH')).mkdir(parents=True, exist_ok=True)

    log_file = open(Path(os.environ.get('TRAIN_LOG_PATH'), 'train.log'), 'a')
    writer = SummaryWriter(os.environ.get('TRAIN_TF_EVENTS_PATH'))
    
    data_path = os.environ.get('TRAIN_DATA_PATH')

    args = get_args()
    dataset = MyDataset(data_path, args)
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [0.9, 0.1])
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16, collate_fn=dataset.collate_fn, pin_memory=True
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=dataset.collate_fn
    )
    usernum, itemnum = dataset.usernum, dataset.itemnum
    feat_statistics, feat_types = dataset.feat_statistics, dataset.feature_types

    model = BaselineModel(usernum, itemnum, feat_statistics, feat_types, args).to(args.device)

    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except Exception:
            pass

    model.pos_emb.weight.data[0, :] = 0
    model.item_emb.weight.data[0, :] = 0
    model.user_emb.weight.data[0, :] = 0

    for k in model.sparse_emb:
        model.sparse_emb[k].weight.data[0, :] = 0

    epoch_start_idx = 1

    if args.state_dict_path is not None:
        try:
            model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
            tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6 :]
            epoch_start_idx = int(tail[: tail.find('.')]) + 1
        except: 
            print('failed loading state_dicts, pls check file path: ', end="")
            print(args.state_dict_path)
            raise RuntimeError('failed loading state_dicts, pls check file path!')

    bce_criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

# ====================================== Resume Settings ====================================
    global_step = 0
    resume_candidate = None
    if args.resume_path is not None:
        resume_candidate = Path(args.resume_path)
    elif args.resume:
        latest = _ckpt_dir() / "latest.pt"
        if latest.exists():
            resume_candidate = latest
        else:
            ckpts = _list_ckpts()
            if ckpts:
                resume_candidate = ckpts[-1]

    if resume_candidate is not None and resume_candidate.exists():
        try:
            last_epoch, global_step = _load_ckpt(
                model=model, optimizer=optimizer, device=args.device, ckpt_path=resume_candidate
            )
            epoch_start_idx = last_epoch + 1
            print(f"Resumed from {str(resume_candidate)} (epoch={last_epoch}, global_step={global_step})")
        except Exception as e:
            print(f"Failed to resume from {str(resume_candidate)}: {e}")
            raise
    elif args.state_dict_path is not None:
        # backward compatibility: model-only weights
        try:
            model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
            tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6 :]
            epoch_start_idx = int(tail[: tail.find('.')]) + 1
        except:
            print('failed loading state_dicts, pls check file path: ', end="")
            print(args.state_dict_path)
            raise RuntimeError('failed loading state_dicts, pls check file path!')
        
# ======================================= Train Process ====================================

    best_val_ndcg, best_val_hr = 0.0, 0.0
    best_test_ndcg, best_test_hr = 0.0, 0.0
    T = 0.0
    t0 = time.time()
    
    print("Start training")
    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        print(f"======== Epoch {epoch} / {args.num_epochs} ========")
        model.train()
        # print("Model Training...")
        if args.inference_only:
            break
        for step, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
            # print("Start processing batch...")
            seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat = batch
            
            seq = seq.to(args.device, non_blocking=True)
            pos = pos.to(args.device, non_blocking=True)
            neg = neg.to(args.device, non_blocking=True)
            token_type = token_type.to(args.device, non_blocking=True)
            next_token_type = next_token_type.to(args.device, non_blocking=True)
            # print("After moving to device, Start Forward...")
            pos_logits, neg_logits = model(
                seq, pos, neg, token_type, next_token_type, seq_feat, pos_feat, neg_feat
            )
            # print("Forward done.")
            pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(
                neg_logits.shape, device=args.device
            )

            optimizer.zero_grad()
            indices = (next_token_type == 1)
            
            loss = bce_criterion(pos_logits[indices], pos_labels[indices])
            loss += bce_criterion(neg_logits[indices], neg_labels[indices])

            if step % 100 == 0:
                log_json = json.dumps(
                    {'global_step': global_step, 'loss': loss.item(), 'epoch': epoch, 'time': time.time()}
                )
                log_file.write(log_json + '\n')
                log_file.flush()
                print(log_json)

            writer.add_scalar('Loss/train', loss.item(), global_step)

            for param in model.item_emb.parameters():
                loss += args.l2_emb * torch.norm(param)
            loss.backward()
            optimizer.step()

            global_step += 1

            # New: periodic checkpoint
            if args.save_every_steps > 0 and (global_step % args.save_every_steps == 0):
                _save_ckpt(model=model, optimizer=optimizer, epoch=epoch, global_step=global_step, args=args)

        

        model.eval()
        valid_loss_sum = 0
        for step, batch in tqdm(enumerate(valid_loader), total=len(valid_loader)):
            seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat = batch
            seq = seq.to(args.device)
            pos = pos.to(args.device)
            neg = neg.to(args.device)
            next_token_type = next_token_type.to(args.device)
            
            # pos_logits, neg_logits = model(
            #     seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat
            # )
            pos_logits, neg_logits = model(
                seq, pos, neg, token_type, next_token_type, seq_feat, pos_feat, neg_feat
            )

            pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(
                neg_logits.shape, device=args.device
            )
            
            indices = (next_token_type == 1)

            loss = bce_criterion(pos_logits[indices], pos_labels[indices])
            loss += bce_criterion(neg_logits[indices], neg_labels[indices])
            valid_loss_sum += loss.item()
        valid_loss_sum /= len(valid_loader)
        writer.add_scalar('Loss/valid', valid_loss_sum, global_step)
        print("Valid loss: {:.4f}".format(valid_loss_sum) + f" at step {global_step}")

        save_dir = Path(os.environ.get('TRAIN_CKPT_PATH'), f"global_step{global_step}.valid_loss={valid_loss_sum:.4f}")
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_dir / "model.pt")

    print("Done")
    writer.close()
    log_file.close()
