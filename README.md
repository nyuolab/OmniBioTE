# OmniBioTE: A Large-Scale Multimodal Biosequence Transformer Model

OmniBioTE is a transformer model designed to capture the complex relationships inherent in biological sequences. Drawing inspiration from the BERT architecture, it supports pretraining on large unlabeled datasets and fine-tuning for various downstream tasks. With multiple tokenization strategies and robust distributed training, OmniBioTE offers a comprehensive solution for multimodal biosequence analysis.

---

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Data Loading and Preprocessing](#data-loading-and-preprocessing)
- [Tokenizer Training](#tokenizer-training)
- [Model Training](#model-training)
  - [Training Scripts and Options](#training-scripts-and-options)
  - [Distributed Training & Checkpointing](#distributed-training--checkpointing)
  - [Command-line Flags](#command-line-flags)
- [Evaluation and Finetuning](#evaluation-and-finetuning)
  - [GUE Evaluation](#gue-evaluation)
  - [TAPE Evaluation](#tape-evaluation)
  - [ProteinGLUE Evaluation](#proteinglue-evaluation)
  - [Additional Evaluation Modules](#additional-evaluation-modules)
    - [GUE Evaluation (LucaOne Version)](#gue-evaluation-lucaone-version)
    - [TAPE Evaluation (LucaOne Version)](#tape-evaluation-lucaone-version)
    - [TAPE Evaluation (ESM Version)](#tape-evaluation-esm-version)
    - [Contact Evaluation (OmniBioTA)](#contact-evaluation-omnibiota)
    - [Contact Evaluation (ESM Version)](#contact-evaluation-esm-version)
    - [Contact Evaluation (LucaOne Version)](#contact-evaluation-lucaone-version)
    - [PDB Contact Evaluation](#pdb-contact-evaluation)
    - [Motif Selectivity Evaluation](#motif-selectivity-evaluation)
    - [Pronab Evaluation](#pronab-evaluation)
    - [Pronab Evaluation (Unimodal)](#pronab-evaluation-unimodal)
    - [Pronab Evaluation (LucaOne)](#pronab-evaluation-lucaone)
  - [Single-Character Evaluation Modules](#single-character-evaluation-modules)
    - [GUE Evaluation (Single-Char)](#gue-evaluation-single-char)
    - [Pronab Evaluation (Single-Char)](#pronab-evaluation-single-char)
    - [ProteinGLUE Evaluation (Single-Char)](#proteinglue-evaluation-single-char)
    - [TAPE Evaluation (Single-Char)](#tape-evaluation-single-char)
- [Additional Modules and Utilities](#additional-modules-and-utilities)
- [Conclusion](#conclusion)

---

## Introduction

OmniBioTE is built to handle the unique characteristics of biological sequences. The model offers two tokenization strategies— a SentencePiece-based byte-pair-encoding model and a single-character tokenizer.

---

## Features

- **Multimodal Support:** Process DNA, RNA, and protein sequences.
- **Flexible Tokenization:** Choose between SentencePiece tokenization and a single-character tokenizer.
- **Robust Data Loading:** A custom data loader employs multithreading and prefetching to efficiently stream sequences from compressed files.
- **Distributed Training:** Integrated support for Distributed Data Parallel (DDP) and FullyShardedDataParallel (FSDP) enables scaling across multiple GPUs.
- **Dynamic Checkpointing:** Automatically resumes training from the latest checkpoint and manages disk space by cleaning up older checkpoints.
- **Comprehensive Evaluation Pipelines:** Evaluate the model on tasks ranging from classification and regression to contact map prediction using specialized evaluation modules.
- **Performance Metrics:** Tools are provided to compute detailed tensor statistics and monitor model performance.

---

## Data Loading and Preprocessing

The data loading pipeline, implemented in [`training/loader.py`](training/loader.py), includes:

- **Multi-Threaded File Preloading:** Uses a background thread with a queue to preload gzip-compressed files.
- **Randomized Chunking:** Splits files into sub-blocks and randomly samples fixed-size chunks for training.
- **Token Filtering:** Applies a configurable offset to token IDs and filters out banned tokens (such as PAD, MASK, EOS).

The custom dataset class, `OmniDataset`, combines data from multiple directories based on specified fractions. It shuffles file lists and streams sequences seamlessly into training batches.

---

## Tokenizer Training

OmniBioTE supports two tokenization approaches:

1. **SentencePiece Tokenization:**  
   Train a [SentencePiece](https://github.com/google/sentencepiece) model to handle long biosequences. Tokenizer training scripts (e.g., `tokenization/tokenize_genbank.py` and `tokenization/tokenize_uniref.py`) and notebooks (e.g., `train_tokenizer.ipynb`) guide the process of building an efficient vocabulary.

2. **Single-Character Tokenization:**  
   A built-in single-character tokenizer (see `train_encoder-single-char.py`) maps each character to a fixed token ID.

Both methods incorporate mechanisms for shifting token IDs by a configurable offset (in order to yield non-overlapping token IDs for two different BPE tokenizers) and banning specific tokens (such as reserved IDs for `<PAD>`, `<MASK>`, and `<EOS>`).

---

## Model Training

Two primary training scripts address different tokenization schemes:

### Training Scripts and Options

- **`train_encoder.py`:**  
  Trains OmniBioTE using BPE tokenizers. This script accepts paths to pre-trained SentencePiece models and adjusts vocabulary offsets and banned tokens accordingly.  
  _Example:_
  ```bash
  torchrun --nnodes=1 --nproc_per_node=4 train_encoder.py \
      --n_head 8 --n_embd 1024 --n_layer 8 \
      --mini_batch_size 2 --batch_size 1024 \
      --lr 0.05 --save_name omnbiote-small \
      --DNA_tokenizer path/to/dna.model --peptide_tokenizer path/to/peptide.model \
      --tokenizer_suffix 8k --train_type mixed
  ```

- **`train_encoder-single-char.py`:**  
  Trains OmniBioTE using the built-in single-character tokenizer. This script directly maps nucleotides and amino acids to token IDs using a fixed vocabulary size.
  _Example:_
  ```bash
  torchrun --nnodes=1 --nproc_per_node=4 train_encoder-single-char.py \
      --n_head 8 --n_embd 1024 --n_layer 8 \
      --mini_batch_size 2 --batch_size 1024 \
      --lr 0.05 --save_name omnbiote-small-single \
      --train_type mixed
  ```

### Distributed Training & Checkpointing

Both training scripts include:
- **Distributed Setup:** Uses `torch.distributed` for multi-GPU training with NCCL/Gloo backends.
- **Gradient Accumulation:** Aggregates gradients when per-GPU batch sizes are small.
- **Dynamic Checkpointing:** Saves checkpoints based on token counts (using `--save_freq`) and resumes training using a `_last_step.txt` file.
- **Learning Rate Scheduling:** Implements a OneCycleLR scheduler with configurable warmup periods and muP scaling for optimizer consistency.

### Command-line Flags

Key training parameters include:
- **Batch & Data Parameters:** `--batch_size`, `--mini_batch_size`, `--ctx_len`, `--base_dir`
- **Model Architecture:** `--n_head`, `--n_embd`, `--n_layer`, `--dropout`, `--position_encoding`
- **Optimization:** `--lr`, `--beta1`, `--beta2`, `--epsilon`, `--weight_decay`, `--force_lr`, `--token_budget`, `--warmup_period`
- **Logging & Checkpointing:** `--test_freq`, `--save_freq`, `--save_name`, `--wandb_project_name`, `--disable_flash`
- **Tokenization Options (SentencePiece only):** `--DNA_tokenizer`, `--peptide_tokenizer`, `--tokenizer_suffix`
- **Other Modes:** `--train_type`, `--FSDP`, `--use_padding`, `--compile`

---

## Evaluation and Finetuning

OmniBioTE includes a comprehensive suite of evaluation pipelines covering a range of downstream tasks. These pipelines support classification, regression, contact map prediction, and more.

### GUE Evaluation

Located in [`evals/gue.py`](evals/gue.py), this module performs full parameter finetuning on the various GUE tasks.

_Run Example:_
```bash
python evals/gue.py --sp_dir path/to/sp.model --model_dir path/to/model.pt --batch_size 32
```

### TAPE Evaluation

The [`evals/TAPE.py`](evals/TAPE.py) module is tailored for protein structure and function tasks:

_Run Example:_
```bash
python evals/TAPE.py --sp_dir path/to/sp.model --model_dir path/to/omnbiote-small.pt --tokenizer_offset 0
```

The tokenizer offset should be set to 2048 for mixed models, since there are 2048 nucleotide tokens and 2048 protein tokens, ordered sequentially.

### ProteinGLUE Evaluation

Located in [`evals/ProteinGLUE.py`](evals/ProteinGLUE.py).

_Run Example:_
```bash
python evals/ProteinGLUE.py --sp_dir path/to/sp.model --model_dir path/to/omnbiote-small.pt --tokenizer_offset 2048 --output_suffix experiment1
```

### Additional Evaluation Modules

We provide the evaluation scripts for several baselines:

#### GUE Evaluation (LucaOne Version)
- **Location:** `evals/gue_lucaone.py`
- **Overview:**  
  An alternative GUE evaluation pipeline that uses a custom `Alphabet` class for tokenization along with the pre-trained LucaOne model from HuggingFace. It performs finetuning with gradient accumulation and OneCycleLR scheduling while computing MCC and weighted F1 scores.
- **Run Example:**
  ```bash
  python evals/gue_lucaone.py --num_accum_steps 1 --batch_size 32 --lr 0.00015625 --embed_lr 0.00015625 --head_lr 0.01 --output_suffix my_experiment
  ```

#### TAPE Evaluation (LucaOne Version)
- **Location:** `evals/TAPE_lucaone.py`
- **Overview:**  
  Adapts the TAPE benchmark for protein tasks using the LucaOne model and a custom `Alphabet` tokenizer. It supports tasks such as secondary structure, remote homology, fluorescence, and stability.
- **Run Example:**
  ```bash
  python evals/TAPE_lucaone.py --batch_size 32 --finetuning_lr 0.0002 --output_suffix my_tape_experiment
  ```

#### TAPE Evaluation (ESM Version)
- **Location:** `evals/TAPE_esm.py`
- **Overview:**  
  Provides an alternative TAPE evaluation pipeline based on Facebook’s ESM models. It integrates native ESM tokenization and supports various model sizes (XS, S, M, L, XL) with a similar finetuning and evaluation routine.
- **Run Example:**
  ```bash
  python evals/TAPE_esm.py --model_size M --batch_size 32 --finetuning_lr 0.0001 --output_suffix esm_eval
  ```

#### Contact Evaluation (OmniBioTE)
- **Location:** `contact_eval.py`
- **Overview:**  
  Trains a protein contact predictor for the TAPE evals using an OmniBioTE model. It processes ProteinNet datasets, adds special markers during tokenization, builds a ResNet-based contact predictor head, and evaluates performance (precision and AUPRC) on medium- and long-range contacts.
- **Run Example:**
  ```bash
  python contact_eval.py --tokenizer_fn path/to/sp.model --model_fn path/to/omnbiota.pt --banned_token 2044 --tokenizer_offset 2048 --wandb_prefix "MyRun" --data_dir /path/to/data
  ```

#### Contact Evaluation (ESM Version)
- **Location:** `contact_eval_esm.py`
- **Overview:**  
  Adapts the contact evaluation procedure for Facebook’s ESM2 models using HuggingFace’s `AutoTokenizer` and `EsmModel`. A ResNet-based contact predictor head is applied to ESM2 embeddings, supporting multiple model sizes with similar training and evaluation setups.
- **Run Example:**
  ```bash
  python contact_eval_esm.py --model_size M --data_dir /path/to/data --num_accumulation_steps 128 --num_epochs 128 --head_dim 128 --num_resnet_blocks 8 --logging
  ```

#### Contact Evaluation (LucaOne Version)
- **Location:** `contact_eval_lucaone.py`
- **Overview:**  
  Implements the contact map prediction using the pre-trained LucaOne model. A custom `Alphabet` class handles tokenization and maps residue-level contacts into token space.
- **Run Example:**
  ```bash
  python contact_eval_lucaone.py --data_dir /path/to/data --num_accumulation_steps 128 --num_epochs 128 --head_dim 128 --num_resnet_blocks 8 --logging
  ```

#### PDB Contact Evaluation
- **Location:** `evals/pdb_contact_eval.py`
- **Overview:**  
  Provides a pipeline for protein contact map prediction using peptide–nucleotide distance data from a PDB-derived JSON file.
- **Run Example:**  
  ```bash
  python evals/pdb_contact_eval.py <model_fn> <name_suffix> <lr> <embed_lr> <head_lr> <tokenizer_offset> <distance_threshold>
  ```

#### Motif Selectivity Evaluation
- **Location:** `evals/motif_selectivity.py`
- **Overview:**  
  Evaluates sequence motif selectivity by comparing wild-type motifs against mutated or randomly generated variants. This module supports dual tokenization for nucleotide and protein sequences, options for introducing mutations at specified rates, and cross-validation with multiple replicates per motif. It logs ΔG (free energy) differences and provides summary statistics across mutation rates.
- **Run Example:**  
  ```bash
  python evals/motif_selectivity.py --genbank_model_path path/to/genbank.model --uniref_model_path path/to/uniref.model --banned_sequences_file path/to/banned.txt --jaspar_data_file path/to/JASPAR2024_CORE_processed.json --model_base_dir path/to/models --folds 10 --replicates 8 --output_suffix my_motif_experiment --mutation_rates 0.05,0.1,0.25,0.5
  ```

#### Pronab Evaluation
- **Location:** `evals/pronab.py`
- **Overview:**  
  Finetunes and evaluates OmniBioTA on pronab/mutation data using custom tokenization for nucleotide and protein sequences. The module evaluates G0 predictions using Pearson correlation and mean absolute error (MAE).
- **Run Example:**  
  ```bash
  python evals/pronab.py --model_fn path/to/model.pt --nucleotide_tokenizer path/to/nuc.model --protein_tokenizer path/to/prot.model --num_accumulation_steps 256 --num_epochs 32 --lr 1e-4 --embed_lr 1e-3 --head_lr 1e-2 --nuc_banned_token 2037 --prot_banned_token 2044 --prot_offset 2048 --save_dir path/to/save --num_splits 10
  ```

#### Pronab Evaluation (Unimodal)
- **Location:** `evals/pronab-unimodal.py`
- **Overview:**  
  A unimodal baseline that uses a single-omic nucleic acid model and a single-omic protein model. It mirrors the training and evaluation process of the standard Pronab Evaluation while processing inputs as a single modality.
- **Run Example:**  
  ```bash
  python evals/pronab-unimodal.py --nuc_model_fn path/to/nuc_model.pt --protein_model_fn path/to/prot_model.pt --nucleotide_tokenizer path/to/nuc.model --protein_tokenizer path/to/prot.model --num_accumulation_steps 256 --num_epochs 32 --lr 1e-4 --embed_lr 1e-3 --head_lr 1e-2 --num_splits 10 --save_dir path/to/save
  ```

#### Pronab Evaluation (LucaOne)
- **Location:** `pronab_lucaone.py`
- **Overview:**  
  Adapts the pronab evaluation framework for the LucaOne model from HuggingFace.
- **Run Example:**  
  ```bash
  python pronab_lucaone.py --num_accumulation_steps 256 --num_epochs 32 --lr 1e-4 --embed_lr 1e-4 --head_lr 1e-2 --num_splits 10 --save_dir path/to/save
  ```

### Single-Character Evaluation Modules

These modules provide evaluation pipelines analogous to their SentencePiece-based counterparts while using OmniBioTE’s built-in single-character tokenization.

#### GUE Evaluation (Single-Char)
- **Location:** `evals/single-char/gue.py`
- **Overview:**  
- **Run Example:**
  ```bash
  python evals/single-char/gue.py --model_dir path/to/model.pt --num_accum_steps 1 --batch_size 32 --lr 0.00015625 --embed_lr 0.00015625 --head_lr 0.01 --output_suffix singlechar_expt
  ```

#### Pronab Evaluation (Single-Char)
- **Location:** `evals/single-char/pronab.py`
- **Overview:**  
- **Run Example:**
  ```bash
  python evals/single-char/pronab.py --model_dir path/to/model.pt --save_dir path/to/save --num_accum_steps 256 --num_epochs 32 --lr 1e-4 --embed_lr 1e-4 --head_lr 1e-2 --num_splits 10
  ```

#### ProteinGLUE Evaluation (Single-Char)
- **Location:** `evals/single-char/proteinGLUE.py`
- **Overview:**  
- **Run Example:**
  ```bash
  python evals/single-char/proteinGLUE.py --model_dir path/to/model.pt --output_suffix singlechar_pg_eval
  ```

#### TAPE Evaluation (Single-Char)
- **Location:** `evals/single-char/TAPE.py`
- **Overview:**  
- **Run Example:**
  ```bash
  python evals/single-char/TAPE.py --model_dir path/to/model.pt --batch_size 32 --finetuning_lr 1e-4 --output_suffix singlechar_tape_eval
  ```

---

## Additional Modules and Utilities

- **Metrics Utilities:**  
  The [`training/metrics.py`](training/metrics.py) file includes functions to compute tensor statistics (mean, std, L1, L2, min, max, norm) for debugging and monitoring training progress.
  
- **Attention Masking Functions:**  
  Custom JIT-compiled functions in the evaluation modules create block-diagonal attention masks based on `<EOS>` positions and ensure that padded regions are correctly masked.
  
- **Tokenizer Wrappers:**  
  Lightweight wrappers adjust token IDs (via offsets) and filter out banned tokens to ensure consistency between training and evaluation.

---

## Conclusion

OmniBioTE is a versatile transformer model tailored to the challenges of biological sequence analysis. It features robust data loading, flexible tokenization, scalable training infrastructure, and comprehensive evaluation pipelines covering a wide range of bioinformatics tasks. From pretraining on extensive sequence datasets to fine-tuning on domain-specific tasks such as protein–nucleic acid binding, contact prediction, and motif selectivity, OmniBioTE provides a complete, integrated solution for modern biosequence modeling.

Enjoy exploring the capabilities of OmniBioTE as you train and evaluate your models.