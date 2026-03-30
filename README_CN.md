中文 | [English](README.md)

# 2025年腾讯广告算法大赛Baseline

## 1. 简介

本项目是2025年腾讯广告算法大赛的开源Baseline，架构基于SASRec，使用Transformer结构建模用户的历史行为序列。该Baseline旨在为参赛者提供一个高效、易用的起点，帮助选手快速上手并进行模型开发和优化。

## 2. 目录结构

``` 
├── README.md
├── faiss-based-ann
|   ├── resources
|   │   ├── embedding.fbin
|   │   └── id.u64bin
│   ├── main.cc
│   └── CMakeLists.txt
├── dataset.py
├── main.py
├── model.py
├── infer.py
├── eval.py
├── model_rqvae.py
└── requirements.txt
```

## 3. 安装依赖

```bash
pip install -r requirements.txt
```

## 4. 使用方法

### 4.1 模型训练

在 `main.py` 中自行配置:

``` python
run_id = "Your_run_id"
# 都是文件夹名
os.environ["TRAIN_LOG_PATH"] = "Your_train_log_path"
os.environ["TRAIN_TF_EVENTS_PATH"] = "Your_train_tf_events_path"
os.environ["TRAIN_CKPT_PATH"] = "Your_train_ckpt_path"
# global dataset
os.environ["TRAIN_DATA_PATH"] = "Your_train_data_path" 
```

训练命令:

``` bash
python main.py
```

### 4.2 模型推理

注意，在推理之前，需要自行配置 `faiss` 以支持向量检索，一种可能的配置(在 `conda` 环境下)：

- 安装 `faiss`:
    - 通过conda安装已经编译好的so库文件和头文件，参考：https://github.com/facebookresearch/faiss/blob/v1.9.0/INSTALL.md#step-1-invoking-cmake

    ``` bash
        # CPU-only version
        conda install -c pytorch faiss-cpu
    ```

- 安装 `gflags`:
    ``` bash
        conda install anaconda::gflags
    ```

- 构建检索应用程序
    ``` bash
        cd faiss-based-ann
        mkdir build & cd build
        #如果CONDA_PREFIX变量没有值，这一步会报错;
        cmake ..
        make
    ```
    该脚本运行后，会在faiss-based-ann目录下生成一个faiss_demo的可执行文件

在 `eval.py` 中自行配置:

``` python
# 都是文件夹名
os.environ['EVAL_RESULT_PATH'] = 'Your_eval_result_path'

os.environ['EVAL_DATA_PATH'] = 'Your_eval_data_path' 

os.environ["MODEL_OUTPUT_PATH"] = "Your_model_output_path"
```

推理命令:

``` bash
python eval.py
```


### 4.4 可能拓展

`model_rqvae.py` 是一个基于RQ-VAE的模型实现，参赛者可以根据需要选择使用该模型基于比赛提供的 Multimodal-Embedding 训练 Semantic ID。
