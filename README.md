[中文](README_CN.md) | English

# 2025 Tencent Ads Algorithm Competition Baseline

## 1. Introduction

This project is the Baseline for the 2025 Tencent Ads Algorithm Competition. It is built on the SASRec architecture, using a Transformer structure to model users' historical behavior sequences. This Baseline aims to provide participants with an efficient and easy-to-use starting point to quickly get started with model development and optimization.

## 2. Directory Structure

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

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 4. Usage

### 4.1 Model Training

Configure the following in `main.py`:

``` python
run_id = "Your_run_id"
# All are directory names
os.environ["TRAIN_LOG_PATH"] = "Your_train_log_path"
os.environ["TRAIN_TF_EVENTS_PATH"] = "Your_train_tf_events_path"
os.environ["TRAIN_CKPT_PATH"] = "Your_train_ckpt_path"
# global dataset
os.environ["TRAIN_DATA_PATH"] = "Your_train_data_path" 
```

Training command:

``` bash
python main.py
```

### 4.2 Model Inference

Note: Before inference, you need to configure `faiss` for vector retrieval. A possible setup (under a `conda` environment):

- Install `faiss`:
    - Install pre-compiled shared libraries and headers via conda. Reference: https://github.com/facebookresearch/faiss/blob/v1.9.0/INSTALL.md#step-1-invoking-cmake

    ``` bash
        # CPU-only version
        conda install -c pytorch faiss-cpu
    ```

- Install `gflags`:
    ``` bash
        conda install anaconda::gflags
    ```

- Build the retrieval application:
    ``` bash
        cd faiss-based-ann
        mkdir build & cd build
        # This step will fail if the CONDA_PREFIX variable is not set
        cmake ..
        make
    ```
    After running the script, a `faiss_demo` executable will be generated in the `faiss-based-ann` directory.

Configure the following in `eval.py`:

``` python
# All are directory names
os.environ['EVAL_RESULT_PATH'] = 'Your_eval_result_path'

os.environ['EVAL_DATA_PATH'] = 'Your_eval_data_path' 

os.environ["MODEL_OUTPUT_PATH"] = "Your_model_output_path"
```

Inference command:

``` bash
python eval.py
```


### 4.4 Possible Extensions

`model_rqvae.py` is a model implementation based on RQ-VAE. Participants can choose to use this model to train Semantic IDs based on the Multimodal-Embedding provided in the competition.
