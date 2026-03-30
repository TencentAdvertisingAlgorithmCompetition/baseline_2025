import json
import os
import shutil
# import sys
import time
from pathlib import Path

# sys.path.append(os.environ.get("EVAL_INFER_PATH"))

from infer import infer


def delete_all_except(directory, exclude_file):
    """
    删除指定目录中除指定文件外的所有文件和子目录

    :param directory: 要清理的目录路径
    :param exclude_file: 要保留的文件名
    """
    # 遍历目录中的所有内容
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)

        # 跳过要保留的文件
        if item == exclude_file and os.path.isfile(item_path):
            continue

        try:
            # 如果是文件，删除文件
            if os.path.isfile(item_path):
                os.remove(item_path)
            # 如果是目录，删除整个目录树
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
        except Exception as e:
            print(e)


if __name__ == '__main__':

    # os.environ['EVAL_RESULT_PATH'] = '/apdcephfs_szgm/share_303492287/ryanylsun/TencentGR/competition_test/python/results'
    os.environ['EVAL_RESULT_PATH'] = '/apdcephfs_szgm/share_303492287/ryanylsun/TencentGR/competition_test/python/epoch2_results'

    os.environ['EVAL_DATA_PATH'] = '/apdcephfs_szgm/share_303492287/ryanylsun/TencentGR/second/second'
    
    os.environ["MODEL_OUTPUT_PATH"] = "/apdcephfs_szgm/share_303492287/ryanylsun/TencentGR/competition_test/python/checkpoints/global_step35648.valid_loss=0.0959"
    # os.environ['EVAL_RESULT_PATH'] = '/apdcephfs_szgm/share_303492287/ryanylsun/TencentGR/competition_test/python/epoch1_results'

    os.makedirs(os.environ.get('EVAL_RESULT_PATH'), exist_ok=True)


    result = {}

    t0 = time.time()
    top10s, user_list = infer()
    t1 = time.time()

    result['time'] = t1 - t0
    result['top10s'] = top10s
    result['user'] = user_list
    

    retrieved_less_10 = sum(1 for x in top10s if len(x) < 10)
    if retrieved_less_10 > 0:
        print(f'Warning: {retrieved_less_10 / len(top10s):.3f} test samples matched less than 10 results')

    with open(Path(os.environ.get('EVAL_RESULT_PATH'), "result.json"), 'w') as f:
        json.dump(result, f)
    directory_path = os.environ.get('EVAL_RESULT_PATH')
    delete_all_except(directory_path, 'result.json')
