# Lets-develop-an-efficient-LLM
The goal of this repository is to build an efficient LLM based on Andrej Karpathys "build-nanogpt" (repo: https://github.com/karpathy/build-nanogpt). Ensure to try staying at one file (see train_gpt2.py) and ask me for compute resources. I can rent H100, A100 A40, RTX4090 ... GPUs for a short time (few hours) depending on the importance.  

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
python finweb.py
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

---

<details>
<summary>📌 Note</summary>
Ensure you have the necessary hardware and CUDA setup for GPU training. Adjust the number of GPUs in the multi-GPU command as needed.
</details>
