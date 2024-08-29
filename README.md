# OmniBioTE: A Large-Scale Multimodal Biosequence Transformer Model

## Table of Contents
- [Introduction](#introduction)
- [Using a Pretrained Model](#using-a-pretrained-model)
- [Downloading and Preprocessing Data](#downloading-and-preprocessing-data)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Conclusion](#conclusion)

## Introduction
OmniBioTE is a large-scale multimodal biosequence transformer model that is designed to capture the complex relationships in biological sequences such as DNA, RNA, and proteins. The model is based on the BERT architecture and is adapted to handle the unique characteristics of biosequences.

## Requirements
```
mup==1.0.0
numpy==1.24.4
scikit-learn==1.3.2
scipy==1.10.1
sentencepiece==0.2.0
torch==2.2.1
tqdm==4.66.2
```

## Using a Pretrained Model
After loading a pretrained model, the `encode` method can be used to generate embeddings for a given sequence. The `encode` method takes in a a `torch.LongTensor` of shape `(b, t)` where `b` is the batch dimension and `t` is the token dimension. It returns a `torch.FloatTensor` of shape `(b, n_embd)` where `n_embd` is the embedding dimension of the model. Additionally, a `method` parameter can be passed to the `encode` method to specify the method used to generate the embeddings. The available methods are:
- `mean`: Returns the mean of the embeddings across the token dimension
- `first`: Returns the first token's embedding
- `last`: Returns the last token's embedding
- `max`: Returns the maximum value of the embeddings across the token dimension
- `all`: Returns all of the embeddings across the token dimension

### Example Usage
```python
import torch
from model import OmniBioTA, OmniBioTAConfig

model = torch.load("omnibiote-small.pt", map_location="cuda").to(device) # Load the pretrained model
model.eval() # Set the model to evaluation mode

sequence = torch.randint(0, 100, (1, 1024)).to(device) # Generate a random sequence of length 1024
embeddings = model.encode(sequence, method="mean") # Generate embeddings for the sequence
```

## Downloading and Preprocessing Data
Before training the model, it is necessary to download and preprocess the biological sequence data. The provided scripts `download_genbank.py`, `preprocess_genbank.py`, `preprocess_uniprot.py` facilitate the downloading of GenBank sequences and preprocessing of UniProt sequences, respectively.

### Example Usage
```bash
python preprocessing/download_genbank.py
python preprocessing/preprocess_genbank.py
```

### Data Sources
Nucleic acid data is sourced from [GenBank](https://ftp.ncbi.nlm.nih.gov/genbank/), while peptide data is sourced from [UniProt100](https://www.uniprot.org/help/uniref).


## Model Training
The model training is carried out using the `train_encoder.py` script. The training procedure includes distributed training across multiple GPUs, gradient accumulation (to increase throughput by reducing parameter sync ops), and a number of optimization/stability techniques like µP, weight decay, batch ramp, learning rate decay, and more.

### Example Usage
```bash
torchrun --nnodes=1 --nproc_per_node=4 train_encoder.py --n_head 8 --n_embd 1024 --n_layer 8 --mini_batch_size 2 --lr 0.05 --save_name omnbiote-small
```

The full list of flags and options for the training scripts is as follows:
```
--mini_batch_size: The batch size used for gradient accumulation, and the batch size for each process. Essentially the batch size per GPU at each accumulation step.
--batch_size: The total batch size across all nodes and processes. This, in tandem with `mini_batch_size` essentially determines the number of gradient accumulation steps taken. For example if set to 1024 and there are 4 nodes and 4 processes per node with a mini_batch_size of 2, then each process will automatically do 32 accumulation steps to reach the target total batch size of 1024.
--n_head: The number of attention heads in the model
--n_embd: The embedding dimension of the model
--n_layer: The number of transformer blocks in the model
--ctx_len: The context length of the model
--dropout: The dropout probability used during training
--lr: The learning rate (scaled by muP, unless --force_lr is set)
--beta1: The beta1 parameter for AdamW
--beta2: The beta2 parameter for AdamW
--epsilon: The epsilon parameter for AdamW
--weight_decay: The weight decay parameter for AdamW
--token_budget: The number of tokens to train on (this is only approximate, since our batch ramp slightly decreases the total number of tokens trained on).
--test_freq: The number of tokens to train for between tests
--save_freq: The number of tokens to train for between checkpoints
--save_name: The prefix name to save the model as
--disable_flash: Whether to disable flash attention
--wandb_project_name: The name of the wandb project to log to
--base_dir: The base directory for the training and validation data
--force_lr: Whether to override muP's learning rate scaling. Use this if you want to use a learning rate that is not scaled by muP (not recommended).
```

---

## Additional Notes
- The provided scripts are part of a larger workflow and may need to be adapted to fit into different computational environments or data pipelines.
- For more detailed instructions and information, please refer to the documentation and comments within each script.
