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
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from scipy.stats import spearmanr
from transformers import AutoModel

device = "cuda:0" if torch.cuda.is_available() else "cpu"
BASE_DIR = "../datasets/TAPE/data"

import itertools
from typing import Sequence, List

class Alphabet(object):
    def __init__(
            self,
            standard_toks: Sequence[str] = ['1', '2', '3', '4', '5', 'L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D', 'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C', 'X', 'B', 'U', 'Z', 'O', 'J', '.', '-', '*'],
            prepend_toks: Sequence[str] = ['[PAD]', '[UNK]'],
            append_toks: Sequence[str] = ['[CLS]', '[SEP]', '[MASK]'],
            prepend_bos: bool = True,
            append_eos: bool = True
    ):
        self.standard_toks = list(standard_toks)
        self.prepend_toks = list(prepend_toks)
        self.append_toks = list(append_toks)
        self.prepend_bos = prepend_bos
        self.append_eos = append_eos

        self.all_toks = list(self.prepend_toks)
        self.all_toks.extend(self.append_toks)
        self.all_toks.extend(self.standard_toks)

        self.tok_to_idx = {tok: i for i, tok in enumerate(self.all_toks)}

        self.unk_idx = self.tok_to_idx["[UNK]"]
        self.padding_idx = self.get_idx("[PAD]")
        self.pad_token_id = self.padding_idx
        self.cls_idx = self.get_idx("[CLS]")
        self.mask_idx = self.get_idx("[MASK]")
        self.eos_idx = self.get_idx("[SEP]")
        self.all_special_tokens = prepend_toks + append_toks
        self.all_special_token_idx_list = [self.tok_to_idx[v] for v in self.all_special_tokens]
        self.unique_no_split_tokens = self.all_toks
        self.vocab_size = self.__len__()

    def __len__(self):
        return len(self.all_toks)

    def get_idx(self, tok):
        return self.tok_to_idx.get(tok, self.unk_idx)

    def get_tok(self, ind):
        return self.all_toks[ind]

    def to_dict(self):
        return self.tok_to_idx.copy()

    @classmethod
    def from_predefined(cls, name: str):
        if name.lower() == "prot":
            standard_toks = ['L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D', 'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C', 'X', 'B', 'U', 'Z', 'O', 'J', '.', '-', '*']
        elif name.lower() == "gene":
            standard_toks = ['1', '2', '3', '4', '5', '.', '-', '*']
        elif name.lower() in ["gene_prot", "prot_gene"]:
            standard_toks = ['1', '2', '3', '4', '5', 'L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D', 'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C', 'X', 'B', 'U', 'Z', 'O', 'J', '.', '-', '*']
        else:
            raise Exception("Not support tokenizer name: %s" % name)

        prepend_toks = ['[PAD]', '[UNK]']
        append_toks = ['[CLS]', '[SEP]', '[MASK]']
        prepend_bos = True
        append_eos = True

        return cls(standard_toks, prepend_toks, append_toks, prepend_bos, append_eos)

    @classmethod
    def from_pretrained(cls, dir_path):
        import os, pickle
        return pickle.load(open(os.path.join(dir_path, "alphabet.pkl"), "rb"))

    def save_pretrained(self, save_dir):
        import os, pickle
        with open(os.path.join(save_dir, "alphabet.pkl"), 'wb') as outp:
            pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)

    def _tokenize(self, text) -> str:
        return text.split()

    def tokenize(self, text, **kwargs) -> List[str]:
        def split_on_token(tok, text):
            result = []
            split_text = text.split(tok)
            for i, sub_text in enumerate(split_text):
                if i < len(split_text) - 1:
                    sub_text = sub_text.rstrip()
                if i > 0:
                    sub_text = sub_text.lstrip()

                if i == 0 and not sub_text:
                    result.append(tok)
                elif i == len(split_text) - 1:
                    if sub_text:
                        result.append(sub_text)
                    else:
                        pass
                else:
                    if sub_text:
                        result.append(sub_text)
                    result.append(tok)
            return result

        def split_on_tokens(tok_list, text):
            if not text.strip():
                return []
            tokenized_text = []
            text_list = [text]
            for tok in tok_list:
                tokenized_text = []
                for sub_text in text_list:
                    if sub_text not in self.unique_no_split_tokens:
                        tokenized_text.extend(split_on_token(tok, sub_text))
                    else:
                        tokenized_text.append(sub_text)
                text_list = tokenized_text

            return list(
                itertools.chain.from_iterable(
                    (
                        self._tokenize(token)
                        if token not in self.unique_no_split_tokens
                        else [token]
                        for token in tokenized_text
                    )
                )
            )

        no_split_token = self.unique_no_split_tokens
        tokenized_text = split_on_tokens(no_split_token, text)
        return tokenized_text

    def EncodeAsIds(self, text):
        return [self.tok_to_idx[tok] for tok in self.tokenize(text)]
    
    def DecodeAsIds(self, token_ids: Sequence[int], skip_special_tokens: bool = False) -> str:
        """
        Convert a sequence of token IDs back into a string.
        
        Args:
            token_ids (Sequence[int]): List (or sequence) of token ids.
            skip_special_tokens (bool): Whether or not to filter out special tokens.
        
        Returns:
            str: The decoded string.
        """
        # Convert IDs back to tokens.
        tokens = [self.get_tok(i) for i in token_ids]
        
        # Optionally remove any special tokens (e.g. [PAD], [UNK], [CLS], [SEP], [MASK]).
        if skip_special_tokens:
            tokens = [t for t in tokens if t not in self.all_special_tokens]
        
        # Heuristic for detokenization:
        # If any token looks like a special token (starts with '[' and ends with ']'),
        # join all tokens with a space. Otherwise (for instance, if all tokens are single characters),
        # join them with no delimiter.
        if any(token.startswith('[') and token.endswith(']') for token in tokens):
            decoded = "".join(tokens)
        else:
            decoded = "".join(tokens)
            
        return decoded

# ---------------------  DATA LOADING  ---------------------

def load_secondary_structure(split, base_dir=BASE_DIR):
    """
    Loads secondary structure data (ss3 and ss8).
    The valid `split` values are:
      - 'train', 'valid', 'casp12', 'cb513', 'ts115'.
    Returns:
      sequences, ss3_labels, ss8_labels (lists of equal length)
    """
    filename = os.path.join(base_dir, f"secondary_structure", f"secondary_structure_{split}.json")
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
    filename = os.path.join(base_dir, f"remote_homology", f"remote_homology_{split}.json")
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
    filename = os.path.join(base_dir, f"fluorescence", f"fluorescence_{split}.json")
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
    filename = os.path.join(base_dir, f"stability", f"stability_{split}.json")
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
    tokenized_seq = sp.EncodeAsIds("[CLS]" + sequence + "[SEP]")
    return tokenized_seq, label, None, len(sequence)


def format_sequence_and_single_label(sequence, label, sp):
    """
    Tokenizes the sequence but expects a single integer label for the whole sequence
    (e.g., remote homology classification).
    """
    tokenized_seq = sp.EncodeAsIds("[CLS]" + sequence + "[SEP]")
    return tokenized_seq, label, None, len(sequence)


def format_sequence_and_value(sequence, label, sp):
    """
    Similar to `format_sequence_and_label`, but aggregates a continuous value (mean) over sub-tokens.
    E.g., for secondary structure continuous label or other tasks requiring per-residue regression.
    """
    tokenized_seq = sp.EncodeAsIds("[CLS]" + sequence + "[SEP]")
    return tokenized_seq, label, None, len(sequence)


def format_sequence_and_single_value(sequence, label, sp):
    """
    Tokenizes a sequence for tasks that have a single numeric value per entire sequence
    (e.g., fluorescence or stability).
    """
    tokenized_seq = sp.EncodeAsIds("[CLS]" + sequence + "[SEP]")
    return tokenized_seq, label, None, len(sequence)


def process_data(sp, sequences, target, format_func):
    """
    Prepares data by tokenizing each sequence
    and collecting corresponding label/target arrays.

    Args:
        sequences (list[str]): Input protein sequences or None.
        target (list): Corresponding labels or numeric values.
        format_func (callable): One of the format_sequence_* functions.

    Returns:
        (tokenized, targets) or (None, None) if input was None.
    """
    if sequences is None:
        return None, None

    sequences_tokenized, targets_out = [], []
    for i in tqdm(range(len(sequences)), desc="Tokenizing"):
        tk_seq, tk_label, _, _ = format_func(sequences[i], target[i], sp)
        sequences_tokenized.append(tk_seq)
        targets_out.append(tk_label)

    return sequences_tokenized, targets_out


def get_training_sets(task_name, sp, format_func):
    """
    Loads train/val/test from load_task, then calls process_data on each.
    Also normalizes regression tasks (fluorescence, stability) to mean=0, std=1.

    Returns: 
      X_train, Y_train, X_val, Y_val, X_test, Y_test
    """
    X_tr_raw, Y_tr_raw, X_val_raw, Y_val_raw, X_ts_raw, Y_ts_raw = load_task(task_name)
    X_train, Y_train = process_data(sp, X_tr_raw, Y_tr_raw, format_func)
    X_val, Y_val = process_data(sp, X_val_raw, Y_val_raw, format_func)
    X_test, Y_test = process_data(sp, X_ts_raw, Y_ts_raw, format_func)

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
        sp: HuggingFace tokenizer
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
    X_train, Y_train, X_val, Y_val, _, _ = get_training_sets(task_name, sp, format_func)

    base_model = copy.deepcopy(model)
    base_model.train()

    # Determine output dimension
    if single_target:
        output_dim = 1 if loss_str == 'mse' else (max(Y_train) + 1)
    else:
        # Flatten out all labels for multi-label tasks
        all_labels = [val for sublist in Y_train for val in sublist]
        output_dim = 1 if loss_str == 'mse' else (max(all_labels) + 1)

    head = torch.nn.Linear(
        2560,
        output_dim
    ).to(device).to(dtype)

    # Parameter grouping for different learning rates
    param_groups = [
        {
            "params": [p for n, p in base_model.named_parameters() if "embed" in n],
            "lr": embed_lr
        },
        {
            "params": [p for n, p in base_model.named_parameters() if "embed" not in n],
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
    best_model_sd = copy.deepcopy(base_model.state_dict())
    best_head_sd = copy.deepcopy(head.state_dict())

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
                x_tensor = torch.tensor(x_seq, device=device, dtype=torch.long).unsqueeze(0)
                x_tensor = x_tensor[:, :1280]

                # If single target => use the first token embedding only
                # else => one label per token
                if single_target:
                    embeddings = base_model(input_ids=x_tensor, token_type_ids=torch.ones_like(x_tensor)).hidden_states
                    emb_used = embeddings[:, 0]
                else:
                    embeddings = base_model(input_ids=x_tensor, token_type_ids=torch.ones_like(x_tensor)).hidden_states
                    emb_used = embeddings[:, 1 : -1]

                output = head(emb_used).squeeze(0)

                # Prepare target
                if loss_str == "mse":
                    # For single_target, shape => [1], else [seq_len] => float
                    y_tensor = torch.tensor(y_lab, device=device, dtype=dtype)
                    if single_target:
                        y_tensor = y_tensor.unsqueeze(0)  # => [1]

                    else:
                        y_tensor = y_tensor.unsqueeze(1)  # => [seq_len, 1]
                        y_tensor = y_tensor[:1278]  # 1280 - 2 (CLS, SEP)
                else:
                    # cross_entropy => int labels
                    y_tensor = torch.tensor(y_lab, device=device, dtype=torch.long)
                    if not single_target:
                        y_tensor = y_tensor[:1278]  # 1280 - 2 (CLS, SEP)

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
                    y_lab = Y_val[i]

                    x_tensor = torch.tensor(x_seq, device=device, dtype=torch.long).unsqueeze(0)
                    x_tensor = x_tensor[:, :1280]

                    if single_target:
                        embeddings = base_model(input_ids=x_tensor, token_type_ids=torch.ones_like(x_tensor)).hidden_states
                        emb_used = embeddings[:, 0]
                    else:
                        embeddings = base_model(input_ids=x_tensor, token_type_ids=torch.ones_like(x_tensor)).hidden_states
                        emb_used = embeddings[:, 1 : -1]

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
                        val_truths.extend(y_lab[:1278])  # 1280 - 2 (CLS, SEP)

                val_truths = np.asarray(val_truths)
                val_preds = np.asarray(val_preds)
                val_metric = metric_fn(val_truths, val_preds)

                if val_metric > best_val_metric:
                    best_val_metric = val_metric
                    best_model_sd = copy.deepcopy(base_model.state_dict())
                    best_head_sd = copy.deepcopy(head.state_dict())

    # Load best checkpoint
    base_model.load_state_dict(best_model_sd)
    head.load_state_dict(best_head_sd)
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
        sp (HuggingFace tokenizer): Tokenizer.
        format_func (callable): One of the format_sequence_* funcs.
        metric (str): "SCC" or "ACC".
        loss_str (str): "mse" or "cross_entropy".
        single_target (bool): If True => single numeric or class label.

    Returns:
        float => The computed metric on test set.
    """
    model.eval()
    head.eval()
    _, _, _, _, X_test, Y_test = get_training_sets(task_name, sp, format_func)

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
            if len(X_test[i]) > 1280:
                print(f"Sequence {i} is too long ({len(X_test[i])})")
                continue

            x_seq = X_test[i]
            y_label = Y_test[i]

            x_tensor = torch.tensor(x_seq, device=device, dtype=torch.long).unsqueeze(0)
            x_tensor = x_tensor[:, :1280]

            if single_target:
                emb = model(input_ids=x_tensor, token_type_ids=torch.ones_like(x_tensor)).hidden_states
                emb_used = emb[:, 0]
            else:
                emb = model(input_ids=x_tensor, token_type_ids=torch.ones_like(x_tensor)).hidden_states
                emb_used = emb[:, 1 : -1]

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
                truths_list.extend(y_label[:1278]) # 1280 - 2 (CLS, SEP)

    truths_arr = np.asarray(truths_list)
    preds_arr = np.asarray(preds_list)
    score = metric_fn(truths_arr, preds_arr)

    print(f"{task_name} => {metric}: {score}")
    return score


# ---------------------  MAIN  ---------------------

def main_cli(
    pretraining_epochs=4,
    pretraining_num_accum_steps=4,
    batch_size=32,
    pretraining_lr=1e-3,
    finetuning_lr=2e-4,
    embed_lr=2e-4,
    output_suffix=""
):
    """
    Main entry point for fine-tuning tasks on TAPE dataset.

    Args:
        sp_dir (str): Path to the SentencePiece tokenizer model.
        model_dir (str): Path to the base OmniBioTA model checkpoint.
        tokenizer_offset (int): Offset for token IDs (default=0).
        pretraining_epochs (int): Not used in the original script, can be used if extended.
        pretraining_num_accum_steps (int): Not used in the original script, can be used if extended.
        batch_size (int): Batch size for finetuning.
        pretraining_lr (float): LR for an optional pretraining stage (unused in the final code).
        finetuning_lr (float): LR for the final finetuning stage (default=2e-4).
        output_suffix (str): Suffix for saving output results.
    """

    print(f"Batch size {batch_size}, LR = {finetuning_lr}, embedding LR = {embed_lr}")
    print(f"Output suffix = {output_suffix}")

    # Create tokenizer    
    sp = Alphabet.from_predefined("gene_prot")

    # Load model
    model = AutoModel.from_pretrained("Yuanfei/LucaOne", trust_remote_code=True)
    model.eval()
    model.to(device=device, dtype=torch.bfloat16)

    # Tasks to evaluate
    tasks = ["structure_ss3", "structure_ss8", "remote_homology", "fluorescence", "stability"]
             
    results_ft = {}

    for task_name in tasks:
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
    parser.add_argument("--pretraining_epochs", type=int, default=4, help="(Optional) pretraining epochs.")
    parser.add_argument("--pretraining_num_accum_steps", type=int, default=4, help="(Optional) pretraining accum steps.")
    parser.add_argument("--batch_size", type=int, default=32, help="Finetuning batch size.")
    parser.add_argument("--pretraining_lr", type=float, default=1e-3, help="(Optional) pretraining LR.")
    parser.add_argument("--finetuning_lr", type=float, default=1e-4, help="Finetuning LR.")
    parser.add_argument("--embed_lr", type=float, default=1e-4, help="Embedding LR.")
    parser.add_argument("--output_suffix", type=str, default="", help="Suffix for output files.")
    args = parser.parse_args()

    main_cli(
        pretraining_epochs=args.pretraining_epochs,
        pretraining_num_accum_steps=args.pretraining_num_accum_steps,
        batch_size=args.batch_size,
        pretraining_lr=args.pretraining_lr,
        finetuning_lr=args.finetuning_lr,
        embed_lr=args.embed_lr,
        output_suffix=args.output_suffix
    )
