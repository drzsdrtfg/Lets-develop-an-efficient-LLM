# Lets-develop-an-efficient-LLM
The goal of this repository is to build an efficient LLM based on Andrej Karpathys "build-nanogpt" (repo: https://github.com/karpathy/build-nanogpt). Ensure to try staying at one file (see train_efficientgpt2.py) and ask me for compute resources. I can rent H100, A100 A40, RTX4090 ... GPUs for a short time (few hours) depending on the importance.  

# Training the Model

Follow these steps to train the model:

## 1. Clone the Repository

```bash
git clone https://github.com/drzsdrtfg/Lets-develop-an-efficient-LLM.git
```

## 2. Navigate to the Project Directory

```bash
cd Lets-develop-an-efficient-LLM
```

## 3. Install Dependencies

Install the required packages using pip:

```bash
pip install datasets tiktoken transformers
```

## 4. Prepare the Dataset

Load the dataset and divide it into shards:

```bash
python fineweb.py
```

## 5. Train the Model

### Single GPU Training

To train on a single GPU:

```bash
python train_efficientgpt2.py
```

### Multi-GPU Training

For multi-GPU training (e.g., 8 GPUs):

```bash
torchrun --standalone --nproc_per_node=8 train_efficientgpt2.py
```
# Cuda model training (cudadev; >12.4)
```bash
git clone https://github.com/drzsdrtfg/Lets-develop-an-efficient-LLM.git
cd Lets-develop-an-efficient-LLM
pip install tqdm tiktoken requests datasets transformers
# for me, CUDA 12 (run `nvcc --version`) running on Linux x86_64 Ubuntu 22.04
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install libcudnn9-dev-cuda-12
# "install" cudnn-frontend to ~/
git clone https://github.com/NVIDIA/cudnn-frontend.git

# install MPI (optional, if you intend to use multiple GPUs)
sudo apt install openmpi-bin openmpi-doc libopenmpi-dev
# tokenize the FineWeb dataset 10B tokens sample (takes ~1 hour, get lunch?)
# writes ~19GB of raw GPT-2 tokens to dev/data/fineweb10B
# and ~46GB in ~/.cache/huggingface/datasets/HuggingFaceFW___fineweb
python dev/data/fineweb.py --version 10B
# compile llm.c (mixed precision, with cuDNN flash-attention)
# first compilation is ~1 minute, mostly due to cuDNN
make train_gpt2cu USE_CUDNN=1

# train on a single GPU
./train_gpt2cu \
    -i "dev/data/fineweb10B/fineweb_train_*.bin" \
    -j "dev/data/fineweb10B/fineweb_val_*.bin" \
    -o log124M \
    -e "d12" \
    -b 64 -t 1024 \
    -d 524288 \
    -r 1 \
    -z 1 \
    -c 0.1 \
    -l 0.0006 \
    -q 0.0 \
    -u 700 \
    -n 5000 \
    -v 250 -s 20000 \
    -h 1

# if you have multiple GPUs (e.g. 8), simply prepend the mpi command, e.g.:
# mpirun -np 8 ./train_gpt2cu \ ... (the rest of the args are same)
```
---

<details>
<summary>📌 Note</summary>
Ensure you have the necessary hardware and CUDA setup for GPU training. Adjust the number of GPUs in the multi-GPU command as needed.
</details>
