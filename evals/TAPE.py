import os
import sys
import copy
import json
import re
import pickle
import random
import argparse

import numpy as np
import torch
import sentencepiece as spm
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from scipy.stats import spearmanr

sys.path.insert(0, "../training")
from loader import EOS_TOKEN
from model import OmniBioTA

device = "cuda:0" if torch.cuda.is_available() else "cpu"
BASE_DIR = "../datasets/TAPE/data"


# ---------------------  TOKENIZER WRAPPER  ---------------------

class TokenizerWrapper:
    """
    Thin wrapper around a SentencePiece processor that applies:
    - An offset to every token ID.
    - Banning (removal) of specified tokens from the output.
    """
    def __init__(self, sp_processor, tokenizer_offset, banned_tokens):
        self.sp = sp_processor
        self.tokenizer_offset = tokenizer_offset
        self.banned_tokens = banned_tokens
    
    def EncodeAsIds(self, text):
        tokens = self.sp.EncodeAsIds(text)
        # Shift tokens and remove banned ones
        return [t + self.tokenizer_offset for t in tokens if t not in self.banned_tokens]
    
    def DecodeIds(self, token_ids):
        if not isinstance(token_ids, list):
            token_ids = [token_ids]
        # Reverse the offset
        return self.sp.DecodeIds([t - self.tokenizer_offset for t in token_ids])


# ---------------------  DATA LOADING  ---------------------

def load_secondary_structure(split, base_dir=BASE_DIR):
    """
    Loads secondary structure data (ss3 and ss8).
    The valid `split` values are:
      - 'train', 'valid', 'casp12', 'cb513', 'ts115'.
    Returns:
      sequences, ss3_labels, ss8_labels (lists of equal length)
    """
    filename = os.path.join(base_dir, f"secondary_structure/secondary_structure_{split}.json")
    with open(filename, "r") as f:
        data = json.load(f)

    sequences, ss3_labels, ss8_labels = [], [], []
    for item in data:
        sequences.append(item["primary"])
        ss3_labels.append(item["ss3"])
        ss8_labels.append(item["ss8"])

    assert len(sequences) == len(ss3_labels) == len(ss8_labels)
    return sequences, ss3_labels, ss8_labels


def load_remote_homology(split, base_dir=BASE_DIR):
    """
    Loads remote homology data.
    Valid `split` values:
      - 'train', 'valid', 'test_fold_holdout', 'test_family_holdout', 'test_superfamily_holdout'.
    Returns:
      sequences, class_labels (fold labels).
    """
    filename = os.path.join(base_dir, f"remote_homology/remote_homology_{split}.json")
    with open(filename, "r") as f:
        data = json.load(f)

    sequences, class_labels = [], []
    for item in data:
        sequences.append(item["primary"])
        class_labels.append(item["fold_label"])

    assert len(sequences) == len(class_labels)
    return sequences, class_labels


def load_fluorescence(split, base_dir=BASE_DIR):
    """
    Loads fluorescence data.
    Valid `split` values: 'train', 'valid', 'test'.
    Returns:
      sequences, log_fluorescence (list of float)
    """
    filename = os.path.join(base_dir, f"fluorescence/fluorescence_{split}.json")
    with open(filename, "r") as f:
        data = json.load(f)

    sequences, log_fluorescence = [], []
    for item in data:
        sequences.append(item["primary"])
        assert len(item["log_fluorescence"]) == 1
        log_fluorescence.append(item["log_fluorescence"][0])

    assert len(sequences) == len(log_fluorescence)
    return sequences, log_fluorescence


def load_stability(split, base_dir=BASE_DIR):
    """
    Loads stability data.
    Valid `split` values: 'train', 'valid', 'test'.
    Returns:
      sequences, stability_scores (list of float)
    """
    filename = os.path.join(base_dir, f"stability/stability_{split}.json")
    with open(filename, "r") as f:
        data = json.load(f)

    sequences, stability_scores = [], []
    for item in data:
        sequences.append(item["primary"])
        assert len(item["stability_score"]) == 1
        stability_scores.append(item["stability_score"][0])

    assert len(sequences) == len(stability_scores)
    return sequences, stability_scores


def load_task(task_name):
    """
    High-level loader for multiple tasks. Depending on `task_name`, it loads:
      - Tertiary structure tasks: structure_ss3 / structure_ss8
      - CASP12, CB513, TS115 (ss3 or ss8)
      - Remote homology tasks
      - Fluorescence
      - Stability
    Returns:
      (train_sequences, train_labels, val_sequences, val_labels, test_sequences, test_labels)
      Some may be None if that task is only for testing or training.
    """
    # Initialize to None
    train_seq, train_lab, val_seq, val_lab = None, None, None, None
    test_seq, test_lab = None, None

    # Structure tasks
    if task_name == "structure_ss3":
        train_seq, train_ss3, _ = load_secondary_structure("train")
        val_seq, val_ss3, _ = load_secondary_structure("valid")
        test_seq, test_lab = None, None

        train_lab = train_ss3
        val_lab = val_ss3

    elif task_name == "structure_ss8":
        train_seq, _, train_ss8 = load_secondary_structure("train")
        val_seq, _, val_ss8 = load_secondary_structure("valid")
        test_seq, test_lab = None, None

        train_lab = train_ss8
        val_lab = val_ss8

    elif task_name == "casp12_ss3":
        test_seq, test_ss3, _ = load_secondary_structure("casp12")
        test_lab = test_ss3

    elif task_name == "casp12_ss8":
        test_seq, _, test_ss8 = load_secondary_structure("casp12")
        test_lab = test_ss8

    elif task_name == "cb513_ss3":
        test_seq, test_ss3, _ = load_secondary_structure("cb513")
        test_lab = test_ss3

    elif task_name == "cb513_ss8":
        test_seq, _, test_ss8 = load_secondary_structure("cb513")
        test_lab = test_ss8

    elif task_name == "ts115_ss3":
        test_seq, test_ss3, _ = load_secondary_structure("ts115")
        test_lab = test_ss3

    elif task_name == "ts115_ss8":
        test_seq, _, test_ss8 = load_secondary_structure("ts115")
        test_lab = test_ss8

    # Remote homology tasks
    elif task_name == "remote_homology":
        train_seq, train_lab = load_remote_homology("train")
        val_seq, val_lab = load_remote_homology("valid")
        test_seq, test_lab = None, None

    elif task_name == "remote_homology_test_fold_holdout":
        test_seq, test_lab = load_remote_homology("test_fold_holdout")

    elif task_name == "remote_homology_test_family_holdout":
        test_seq, test_lab = load_remote_homology("test_family_holdout")

    elif task_name == "remote_homology_test_superfamily_holdout":
        test_seq, test_lab = load_remote_homology("test_superfamily_holdout")

    # Fluorescence
    elif task_name == "fluorescence":
        train_seq, train_lab = load_fluorescence("train")
        val_seq, val_lab = load_fluorescence("valid")
        test_seq, test_lab = load_fluorescence("test")

    # Stability
    elif task_name == "stability":
        train_seq, train_lab = load_stability("train")
        val_seq, val_lab = load_stability("valid")
        test_seq, test_lab = load_stability("test")

    else:
        raise ValueError(f"Unknown task: {task_name}")

    return train_seq, train_lab, val_seq, val_lab, test_seq, test_lab


# ---------------------  SEQUENCE FORMATTING / TOKENIZATION  ---------------------

def format_sequence_and_label(sequence, label, sp):
    """
    Tokenizes a sequence and aggregates label assignments by sub-token.

    Args:
        sequence (str): Protein sequence (e.g., "ACDEFG...").
        label (list[int]): Labels for each amino acid in `sequence`.
        sp (SentencePieceProcessor): Tokenizer.

    Returns:
        (tokenized, label_modes, token_lengths, sequence_length)
    """
    tokenized_seq = sp.EncodeAsIds(sequence)
    token_lens = [len(sp.DecodeIds(token)) for token in tokenized_seq]

    label_modes = []
    ptr = 0
    for i, length in enumerate(token_lens):
        # If unknown token => treat length as 1
        if tokenized_seq[i] == 0:
            length = 1
        sub_labels = label[ptr : ptr + length]
        # Find the most frequent label in this subrange
        label_modes.append(np.bincount(sub_labels).argmax())
        ptr += length

    return tokenized_seq, label_modes, token_lens, len(sequence)


def format_sequence_and_single_label(sequence, label, sp):
    """
    Tokenizes the sequence but expects a single integer label for the whole sequence
    (e.g., remote homology classification).
    """
    tokenized_seq = sp.EncodeAsIds(sequence)
    token_lens = [len(sp.DecodeIds(token)) for token in tokenized_seq]
    return tokenized_seq, label, token_lens, len(sequence)


def format_sequence_and_value(sequence, label, sp):
    """
    Similar to `format_sequence_and_label`, but aggregates a continuous value (mean) over sub-tokens.
    E.g., for secondary structure continuous label or other tasks requiring per-residue regression.
    """
    tokenized_seq = sp.EncodeAsIds(sequence)
    token_lens = [len(sp.DecodeIds(token)) for token in tokenized_seq]

    label_means = []
    ptr = 0
    for i, length in enumerate(token_lens):
        if tokenized_seq[i] == 0:
            length = 1
        sub_vals = label[ptr : ptr + length]
        label_means.append(np.mean(sub_vals))
        ptr += length

    return tokenized_seq, label_means, token_lens, len(sequence)


def format_sequence_and_single_value(sequence, label, sp):
    """
    Tokenizes a sequence for tasks that have a single numeric value per entire sequence
    (e.g., fluorescence or stability).
    """
    tokenized_seq = sp.EncodeAsIds(sequence)
    token_lens = [len(sp.DecodeIds(token)) for token in tokenized_seq]
    return tokenized_seq, label, token_lens, len(sequence)


def process_data(sp, sequences, target, format_func, prefix):
    """
    Prepares data by tokenizing each sequence, adding <protein> prefix and <EOS> at the end,
    and collecting corresponding label/target arrays.

    Args:
        sequences (list[str]): Input protein sequences or None.
        target (list): Corresponding labels or numeric values.
        format_func (callable): One of the format_sequence_* functions.
        prefix (list[int]): Pre-tokenized representation of "<protein>".

    Returns:
        (tokenized, targets) or (None, None) if input was None.
    """
    if sequences is None:
        return None, None

    sequences_tokenized, targets_out = [], []
    for i in tqdm(range(len(sequences)), desc="Tokenizing"):
        tk_seq, tk_label, _, _ = format_func(sequences[i], target[i], sp)
        # Insert prefix + EOS at the end
        seq_prefixed = prefix + tk_seq + [EOS_TOKEN]
        sequences_tokenized.append(seq_prefixed)
        targets_out.append(tk_label)

    return sequences_tokenized, targets_out


def get_training_sets(task_name, sp, format_func, prefix):
    """
    Loads train/val/test from load_task, then calls process_data on each.
    Also normalizes regression tasks (fluorescence, stability) to mean=0, std=1.

    Returns: 
      X_train, Y_train, X_val, Y_val, X_test, Y_test
    """
    X_tr_raw, Y_tr_raw, X_val_raw, Y_val_raw, X_ts_raw, Y_ts_raw = load_task(task_name)
    X_train, Y_train = process_data(sp, X_tr_raw, Y_tr_raw, format_func, prefix)
    X_val, Y_val = process_data(sp, X_val_raw, Y_val_raw, format_func, prefix)
    X_test, Y_test = process_data(sp, X_ts_raw, Y_ts_raw, format_func, prefix)

    # If regression tasks, standardize
    if task_name in ["fluorescence", "stability"]:
        # Flatten the entire Y_train for standardization
        all_y = np.array(Y_train, dtype=object)
        all_y_flat = np.hstack(all_y)
        mean_val, std_val = np.mean(all_y_flat), np.std(all_y_flat) + 1e-9

        # Normalize each
        Y_train = [(arr - mean_val) / std_val for arr in Y_train]
        if Y_val is not None:
            Y_val = [(arr - mean_val) / std_val for arr in Y_val]
        if Y_test is not None:
            Y_test = [(arr - mean_val) / std_val for arr in Y_test]

    return X_train, Y_train, X_val, Y_val, X_test, Y_test


# ---------------------  FINETUNING  ---------------------

def finetune_on_task(
    task_name,
    model,
    sp,
    format_func,
    metric,
    loss_str,
    device,
    dtype=torch.bfloat16,
    num_epochs=4,
    batch_size=1,
    num_accumulation_steps=8,
    lr=1e-4,
    embed_lr=1e-4,
    test_freq=100,
    single_target=False
):
    """
    Finetunes a given model on the specified TAPE-like task.

    Args:
        task_name (str): The task identifier (e.g., "structure_ss3", "remote_homology", "fluorescence", etc.).
        model (torch.nn.Module): The base model to finetune.
        sp (TokenizerWrapper): The SentencePiece-based tokenizer.
        format_func (callable): One of the format_sequence_* functions.
        metric (str): "SCC" for Spearman correlation or "ACC" for accuracy.
        loss_str (str): "mse" or "cross_entropy".
        device (str): The computing device.
        dtype: Torch dtype (e.g., torch.bfloat16).
        num_epochs (int): Number of finetuning epochs.
        batch_size (int): Minibatch size.
        num_accumulation_steps (int): Gradient accumulation steps.
        lr (float): Learning rate for non-embedding layers.
        embed_lr (float): Learning rate for embedding layers.
        test_freq (int): How frequently to evaluate on the validation set.
        single_target (bool): If True, there's a single label per sequence (e.g., remote homology, fluorescence, etc.).

    Returns:
        (base_model, head) after loading the best validation checkpoint.
    """
    prefix = [t for t in sp.EncodeAsIds("<protein>")]
    X_train, Y_train, X_val, Y_val, _, _ = get_training_sets(task_name, sp, format_func, prefix)

    #base_model = copy.deepcopy(model)
    base_model = OmniBioTA(model.config).to(dtype=dtype)
    base_model.load_state_dict(model.state_dict())
    best_model_sd = base_model.cpu().state_dict()
    base_model.to(device=device)
    base_model.train()

    # Determine output dimension
    if single_target:
        output_dim = 1 if loss_str == 'mse' else (max(Y_train) + 1)
    else:
        # Flatten out all labels for multi-label tasks
        all_labels = [val for sublist in Y_train for val in sublist]
        output_dim = 1 if loss_str == 'mse' else (max(all_labels) + 1)

    head = torch.nn.Linear(
        base_model.transformer.wte.weight.shape[-1], 
        output_dim
    ).to(device).to(dtype)

    # Parameter grouping for different learning rates
    param_groups = [
        {
            "params": [p for n, p in base_model.named_parameters() if "wte" in n],
            "lr": embed_lr
        },
        {
            "params": [p for n, p in base_model.named_parameters() if "wte" not in n],
            "lr": lr
        },
        {
            "params": head.parameters(),
            "lr": 1e-2
        }
    ]

    num_steps = int(num_epochs * len(X_train) / (batch_size * num_accumulation_steps))
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[embed_lr, lr, 1e-2],
        total_steps=num_steps,
        pct_start=0.05
    )

    # Setup loss function
    if loss_str == "mse":
        loss_fn = torch.nn.MSELoss()
    elif loss_str == "cross_entropy":
        loss_fn = torch.nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unknown loss function: {loss_str}")

    # Metric function
    if metric == "SCC":
        metric_fn = lambda y, pred: spearmanr(y, pred)[0] if len(y) > 1 else 0.0
    elif metric == "ACC":
        metric_fn = accuracy_score
    else:
        raise ValueError(f"Unknown metric: {metric}")

    best_val_metric = float("-inf")  # For ACC or SCC, higher is better
    best_head_sd = head.state_dict()

    pbar = tqdm(range(num_steps), desc=f"Finetuning {task_name}")
    for step in pbar:
        optimizer.zero_grad()
        total_loss = 0.0
        base_model.train()
        head.train()

        # Accumulate over num_accumulation_steps
        for _ in range(num_accumulation_steps):
            indices = np.random.choice(len(X_train), batch_size, replace=False)
            X_batch = np.array(X_train, dtype=object)[indices]
            Y_batch = np.array(Y_train, dtype=object)[indices]

            for x_seq, y_lab in zip(X_batch, Y_batch):
                # Trim to max length of 1024
                x_seq = x_seq[:1024]
                x_tensor = torch.tensor(x_seq, device=device, dtype=torch.long).unsqueeze(0)
                embeddings = base_model(x_tensor, return_embeddings=True)

                # If single target => use the first token embedding only
                # else => one label per token
                if single_target:
                    emb_used = embeddings[:, 0]
                else:
                    # exclude the <protein> token (index=0) if needed
                    emb_used = embeddings[:, 1 : (1 + len(y_lab))]

                output = head(emb_used).squeeze(0)

                # Prepare target
                if loss_str == "mse":
                    # For single_target, shape => [1], else [seq_len] => float
                    y_tensor = torch.tensor(y_lab, device=device, dtype=dtype)
                    if single_target:
                        y_tensor = y_tensor.unsqueeze(0)  # => [1]
                    else:
                        y_tensor = y_tensor.unsqueeze(1)  # => [seq_len, 1]
                else:
                    # cross_entropy => int labels
                    y_tensor = torch.tensor(y_lab, device=device, dtype=torch.long)

                # Divide by (num_accumulation_steps * batch_size) for gradient accumulation
                loss = loss_fn(output, y_tensor) / (num_accumulation_steps * batch_size)
                loss.backward()
                total_loss += loss.item()

        optimizer.step()
        scheduler.step()

        pbar.set_description(f"Step {step}/{num_steps}, Loss: {total_loss:.4f}")

        # Periodic validation
        if step % max(1, (num_steps // test_freq)) == 0 and X_val is not None:
            base_model.eval()
            head.eval()
            with torch.no_grad():
                val_preds, val_truths = [], []
                for i in range(len(X_val)):
                    x_seq = X_val[i]
                    x_seq = x_seq[:1024]
                    y_lab = Y_val[i]

                    x_tensor = torch.tensor(x_seq, device=device, dtype=torch.long).unsqueeze(0)
                    embeddings = base_model(x_tensor, return_embeddings=True)

                    if single_target:
                        emb_used = embeddings[:, 0]
                    else:
                        emb_used = embeddings[:, 1 : (1 + len(y_lab))]

                    out = head(emb_used).squeeze(0)

                    if loss_str == "mse":
                        val_preds.extend(out.reshape(-1).cpu().tolist())
                    else:
                        # classification => argmax
                        if single_target:
                            val_preds.append(out.argmax(dim=-1).cpu().item())
                        else:
                            val_preds.extend(out.argmax(dim=-1).cpu().numpy().tolist())

                    if single_target:
                        val_truths.append(y_lab)
                    else:
                        val_truths.extend(y_lab)

                val_truths = np.asarray(val_truths)
                val_preds = np.asarray(val_preds)
                val_metric = metric_fn(val_truths, val_preds)

                if val_metric > best_val_metric:
                    best_val_metric = val_metric
                    best_model_sd = base_model.state_dict()
                    best_head_sd = head.state_dict()

    # Load best checkpoint
    base_model.load_state_dict(best_model_sd)
    head.load_state_dict(best_head_sd)

    del optimizer, scheduler
    torch.cuda.empty_cache()

    return base_model, head


def test_model(
    task_name,
    model,
    head,
    sp,
    format_func,
    metric,
    loss_str,
    device,
    dtype=torch.bfloat16,
    single_target=False
):
    """
    Evaluates the given (model + head) on the test set of `task_name`.

    Args:
        task_name (str): The TAPE-like task name.
        model (torch.nn.Module): Model.
        head (torch.nn.Module): Classification/regression head.
        sp (TokenizerWrapper): Tokenizer.
        format_func (callable): One of the format_sequence_* funcs.
        metric (str): "SCC" or "ACC".
        loss_str (str): "mse" or "cross_entropy".
        single_target (bool): If True => single numeric or class label.

    Returns:
        float => The computed metric on test set.
    """
    model.eval()
    head.eval()
    prefix = list(sp.EncodeAsIds("<protein>"))
    _, _, _, _, X_test, Y_test = get_training_sets(task_name, sp, format_func, prefix)

    if X_test is None or Y_test is None:
        print(f"No test set for task {task_name} - skipping test.")
        return None

    if metric == "SCC":
        metric_fn = lambda y, pred: spearmanr(y, pred)[0] if len(y) > 1 else 0.0
    elif metric == "ACC":
        metric_fn = accuracy_score
    else:
        raise ValueError(f"Unknown metric: {metric}")

    preds_list, truths_list = [], []
    with torch.no_grad():
        for i in range(len(X_test)):
            x_seq = X_test[i]
            x_seq = x_seq[:1024]
            y_label = Y_test[i]

            x_tensor = torch.tensor(x_seq, device=device, dtype=torch.long).unsqueeze(0)
            emb = model(x_tensor, return_embeddings=True)

            if single_target:
                emb_used = emb[:, 0]
            else:
                emb_used = emb[:, 1 : (1 + len(y_label))]

            out = head(emb_used).squeeze(0)

            # Predictions
            if loss_str == "mse":
                preds_list.extend(out.reshape(-1).cpu().tolist())
            else:
                if single_target:
                    preds_list.append(out.argmax(dim=-1).cpu().item())
                else:
                    preds_list.extend(out.argmax(dim=-1).cpu().numpy().tolist())

            # Ground truth
            if single_target:
                truths_list.append(y_label)
            else:
                truths_list.extend(y_label)

    truths_arr = np.asarray(truths_list)
    preds_arr = np.asarray(preds_list)
    score = metric_fn(truths_arr, preds_arr)

    print(f"{task_name} => {metric}: {score}")
    return score


# ---------------------  MAIN  ---------------------

def main_cli(
    sp_dir,
    model_dir,
    tokenizer_offset=0,
    banned_token=[2044],
    pretraining_epochs=4,
    pretraining_num_accum_steps=4,
    batch_size=32,
    pretraining_lr=1e-3,
    finetuning_lr=2e-4,
    embed_lr=2e-4,
    output_suffix="",
    memory_efficient=False,
    start_at=None
):
    """
    Main entry point for fine-tuning tasks on TAPE dataset.

    Args:
        sp_dir (str): Path to the SentencePiece tokenizer model.
        model_dir (str): Path to the base OmniBioTA model checkpoint.
        tokenizer_offset (int): Offset for token IDs (default=0).
        banned_token (list[int] or int): Token(s) to exclude from outputs (default=[2044]).
        pretraining_epochs (int): Not used in the original script, can be used if extended.
        pretraining_num_accum_steps (int): Not used in the original script, can be used if extended.
        batch_size (int): Batch size for finetuning.
        pretraining_lr (float): LR for an optional pretraining stage (unused in the final code).
        finetuning_lr (float): LR for the final finetuning stage (default=2e-4).
        output_suffix (str): Suffix for saving output results.
    """
    print(f"Loading tokenizer from {sp_dir}...")
    print(f"Loading model from {model_dir}...")
    print(f"Using tokenizer offset {tokenizer_offset}")
    print(f"Using banned token(s) {banned_token}")
    #print(f"Pretraining (unused here) for {pretraining_epochs} epochs with {pretraining_num_accum_steps} accum.")
    print(f"Batch size {batch_size}, LR = {finetuning_lr}, embedding LR = {embed_lr}")
    print(f"Output suffix = {output_suffix}")

    # Prepare the tokenizer
    sp_base = spm.SentencePieceProcessor()
    sp_base.Load(sp_dir)

    # Convert banned_token into a list properly
    if not isinstance(banned_token, list):
        banned_tokens = [1, 2, banned_token]
    else:
        banned_tokens = [1, 2] + banned_token

    # Wrap tokenizer
    sp = TokenizerWrapper(sp_base, tokenizer_offset, banned_tokens)

    # Load the model
    #model = torch.load(model_dir, map_location=device).to(device).to(torch.bfloat16)
    #model.eval()
    #print(f"Model loaded. #Params = {model.get_num_params()/1e6:.2f}M")
    def recurse_load(m):
        """
        Recursively find the original module if it is wrapped by a
        flash-attention-style `_orig_mod`.
        """
        if hasattr(m, "_orig_mod"):
            return recurse_load(m._orig_mod)
        return m

    # Load pretrained model
    base_model = torch.load(model_dir, map_location="cpu")
    base_model = recurse_load(base_model).to(device="cpu", dtype=torch.bfloat16)
    model_sd = base_model.state_dict()
    config = base_model.config
    if memory_efficient:
        config.memory_efficient = True
    model = OmniBioTA(config).to(device="cpu", dtype=torch.bfloat16)
    model.load_state_dict(model_sd)
    model.to(device=device, dtype=torch.bfloat16)
    model.eval()
    print(f"Num params: {model.get_num_params() / 10**6:.2f}M")

    del base_model

    # Tasks to evaluate
    tasks = ["structure_ss3", "structure_ss8", "remote_homology", "fluorescence", "stability"]
    results_ft = {}

    for task_name in tasks:
        if start_at is not None and start_at not in task_name:
            continue

        start_at = None
        
        if "ss3" in task_name or "ss8" in task_name:
            metric_type = "ACC"
            loss_type = "cross_entropy"
            format_func = format_sequence_and_label
            single_target = False
        elif "remote_homology" in task_name:
            metric_type = "ACC"
            loss_type = "cross_entropy"
            format_func = format_sequence_and_single_label
            single_target = True
        elif "fluorescence" in task_name or "stability" in task_name:
            metric_type = "SCC"
            loss_type = "mse"
            format_func = format_sequence_and_single_value
            single_target = True
        else:
            raise ValueError(f"Unknown task: {task_name}")

        print("---------------------------------------------------------------")
        print(f"Finetuning task '{task_name}'...")

        # Hard-coded epochs, accumulation, etc. in original code
        ft_epochs = 64
        ft_num_accum_steps = 1
        embed_lr = finetuning_lr  # same rate for embedding

        finetuned_model, head = finetune_on_task(
            task_name,
            model,
            sp,
            format_func,
            metric_type,
            loss_type,
            device,
            dtype=torch.bfloat16,
            num_epochs=ft_epochs,
            batch_size=batch_size,
            num_accumulation_steps=ft_num_accum_steps,
            lr=finetuning_lr,
            embed_lr=embed_lr,
            test_freq=100,
            single_target=single_target
        )

        # Test on the same task if it has a test set (fluorescence, stability, etc.)
        # If "ss3/ss8/remote_homology", we do sub-tests
        if "ss3" not in task_name and "ss8" not in task_name and "remote_homology" not in task_name:
            result = test_model(
                task_name,
                finetuned_model,
                head,
                sp,
                format_func,
                metric_type,
                loss_type,
                device,
                dtype=torch.bfloat16,
                single_target=single_target
            )
            results_ft[task_name] = result
            print(f"=> {task_name} => {metric_type}: {result}")

        # Secondary structure sub-tests
        if "ss3" in task_name:
            for subtest in ["casp12_ss3", "cb513_ss3", "ts115_ss3"]:
                result = test_model(
                    subtest,
                    finetuned_model,
                    head,
                    sp,
                    format_func,
                    metric_type,
                    loss_type,
                    device,
                    dtype=torch.bfloat16,
                    single_target=False
                )
                results_ft[subtest] = result
        if "ss8" in task_name:
            for subtest in ["casp12_ss8", "cb513_ss8", "ts115_ss8"]:
                result = test_model(
                    subtest,
                    finetuned_model,
                    head,
                    sp,
                    format_func,
                    metric_type,
                    loss_type,
                    device,
                    dtype=torch.bfloat16,
                    single_target=False
                )
                results_ft[subtest] = result

        # Remote homology sub-tests
        if "remote_homology" in task_name:
            for subtest in [
                "remote_homology_test_fold_holdout",
                "remote_homology_test_family_holdout",
                "remote_homology_test_superfamily_holdout"
            ]:
                result = test_model(
                    subtest,
                    finetuned_model,
                    head,
                    sp,
                    format_func,
                    metric_type,
                    loss_type,
                    device,
                    dtype=torch.bfloat16,
                    single_target=True
                )
                results_ft[subtest] = result

        print("---------------------------------------------------------------")

    # Save results
    out_file = f"TAPE_{output_suffix}_results.csv"
    with open(out_file, "w") as f:
        for t, res in results_ft.items():
            f.write(f"{t},{res}\n")

    print(f"Results saved to {out_file}")


if __name__ == "__main__":
    # If you want to keep fire usage:
    # fire.Fire(main_cli)

    # Or use argparse for a more standard interface:
    parser = argparse.ArgumentParser(description="Finetune OmniBioTA model on TAPE tasks.")
    parser.add_argument("--sp_dir", type=str, help="Path to the SentencePiece model file.")
    parser.add_argument("--model_dir", type=str, help="Path to the pre-trained model checkpoint.")
    parser.add_argument("--tokenizer_offset", type=int, default=0, help="Offset for token IDs.")
    parser.add_argument("--banned_token", nargs="+", type=int, default=[2044], help="List of banned tokens.")
    parser.add_argument("--pretraining_epochs", type=int, default=4, help="(Optional) pretraining epochs.")
    parser.add_argument("--pretraining_num_accum_steps", type=int, default=4, help="(Optional) pretraining accum steps.")
    parser.add_argument("--batch_size", type=int, default=32, help="Finetuning batch size.")
    parser.add_argument("--pretraining_lr", type=float, default=1e-3, help="(Optional) pretraining LR.")
    parser.add_argument("--finetuning_lr", type=float, default=2e-4, help="Finetuning LR.")
    parser.add_argument("--embed_lr", type=float, default=1e-4, help="Embedding LR.")
    parser.add_argument("--output_suffix", type=str, default="", help="Suffix for output files.")
    parser.add_argument("--memory_efficient", action="store_true", help="Use memory-efficient training.")
    parser.add_argument("--start_at", type=str, default=None, help="Start at a specific task.")
    args = parser.parse_args()

    main_cli(
        sp_dir=args.sp_dir,
        model_dir=args.model_dir,
        tokenizer_offset=args.tokenizer_offset,
        banned_token=args.banned_token,
        pretraining_epochs=args.pretraining_epochs,
        pretraining_num_accum_steps=args.pretraining_num_accum_steps,
        batch_size=args.batch_size,
        pretraining_lr=args.pretraining_lr,
        finetuning_lr=args.finetuning_lr,
        embed_lr=args.embed_lr,
        output_suffix=args.output_suffix,
        memory_efficient=args.memory_efficient,
        start_at=args.start_at
    )
