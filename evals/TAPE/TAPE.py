import sys
sys.path.insert(0, '../training')
from loader import EOS_TOKEN, PAD_TOKEN, MASK_TOKEN
import numpy as np
import torch
import sentencepiece as spm
from tqdm import tqdm
import copy
from sklearn.metrics import accuracy_score
from scipy.stats import spearmanr

import pickle
import pandas as pd
import re
import os
import json
import fire

device = "cuda:0" if torch.cuda.is_available() else "cpu"
#BASE_DIR = "/home/sully/OmniBioTA/datasets/TAPE/data"
BASE_DIR = "/gpfs/home/chens59/OmniBioTA/datasets/TAPE/data"

def load_secondary_structure(split, base_dir=BASE_DIR):
    # split can be "train", "valid", "casp12", "cb513", or "ts115"
    # the last three are the test sets
    # returns a tuple of lists: (sequences, ss3_labels, ss8_labels)

    with open(os.path.join(base_dir, f"secondary_structure/secondary_structure_{split}.json"), "r") as f:
        data = json.load(f)
    
    sequences = []
    ss3_labels = []
    ss8_labels = []

    for item in data:
        sequences.append(item["primary"])
        ss3_labels.append(item["ss3"])
        ss8_labels.append(item["ss8"])
    
    assert len(sequences) == len(ss3_labels) == len(ss8_labels)
        
    return sequences, ss3_labels, ss8_labels

def load_remote_homology(split, base_dir=BASE_DIR):
    # split can be "train", "valid", "test_fold_holdout", "test_family_holdout", or "test_superfamily_holdout"
    # returns a tuple of lists: (sequences, fold_label)
    # fold_label has 1195 classes

    with open(os.path.join(base_dir, f"remote_homology/remote_homology_{split}.json"), "r") as f:
        data = json.load(f)
    
    sequences = []
    class_labels = []

    for item in data:
        sequences.append(item["primary"])
        class_labels.append(item["fold_label"])
    
    assert len(sequences) == len(class_labels)
        
    return sequences, class_labels

def load_fluorescence(split, base_dir=BASE_DIR):
    # split can be "train", "valid", "test"
    # returns a tuple of lists: (sequences, log_fluorescence)

    with open(os.path.join(base_dir, f"fluorescence/fluorescence_{split}.json"), "r") as f:
        data = json.load(f)
    
    sequences = []
    log_fluorescence = []

    for item in data:
        sequences.append(item["primary"])
        assert len(item["log_fluorescence"]) == 1
        log_fluorescence.append(item["log_fluorescence"][0])
    
    assert len(sequences) == len(log_fluorescence)
        
    return sequences, log_fluorescence

def load_stability(split, base_dir=BASE_DIR):
    # split can be "train", "valid", "test"
    # returns a tuple of lists: (sequences, stability_score)

    with open(os.path.join(base_dir, f"stability/stability_{split}.json"), "r") as f:
        data = json.load(f)
    
    sequences = []
    stability_scores = []

    for item in data:
        sequences.append(item["primary"])
        assert len(item["stability_score"]) == 1
        stability_scores.append(item["stability_score"][0])
    
    assert len(sequences) == len(stability_scores)
        
    return sequences, stability_scores

def load_task(task):
    if task == "structure_ss3":
        train_sequences, train_ss3, train_ss8 = load_secondary_structure("train")
        val_sequences, val_ss3, val_ss8 = load_secondary_structure("valid")
        test_sequences, test_inference = None, None

        train_inference = train_ss3
        val_inference = val_ss3
    if task == "structure_ss8":
        train_sequences, train_ss3, train_ss8 = load_secondary_structure("train")
        val_sequences, val_ss3, val_ss8 = load_secondary_structure("valid")
        test_sequences, test_inference = None, None

        train_inference = train_ss8
        val_inference = val_ss8

    if task == "casp12_ss3":
        train_sequences, train_inference = None, None
        val_sequences, val_inference = None, None
        test_sequences, test_ss3, test_ss8 = load_secondary_structure("casp12")
        
        test_inference = test_ss3
    if task == "casp12_ss8":
        train_sequences, train_inference = None, None
        val_sequences, val_inference = None, None
        test_sequences, test_ss3, test_ss8 = load_secondary_structure("casp12")
        
        test_inference = test_ss8

    if task == "cb513_ss3":
        train_sequences, train_inference = None, None
        val_sequences, val_inference = None, None
        test_sequences, test_ss3, test_ss8 = load_secondary_structure("cb513")

        test_inference = test_ss3
    if task == "cb513_ss8":
        train_sequences, train_inference = None, None
        val_sequences, val_inference = None, None
        test_sequences, test_ss3, test_ss8 = load_secondary_structure("cb513")

        test_inference = test_ss8

    if task == "ts115_ss3":
        train_sequences, train_inference = None, None
        val_sequences, val_inference = None, None
        test_sequences, test_ss3, test_ss8 = load_secondary_structure("ts115")

        test_inference = test_ss3
    if task == "ts115_ss8":
        train_sequences, train_inference = None, None
        val_sequences, val_inference = None, None
        test_sequences, test_ss3, test_ss8 = load_secondary_structure("ts115")

        test_inference = test_ss8

    if task == "remote_homology":
        train_sequences, train_inference = load_remote_homology("train")
        val_sequences, val_inference = load_remote_homology("valid")
        test_sequences, test_inference = None, None
    if task == "remote_homology_test_fold_holdout":
        train_sequences, train_inference = None, None
        val_sequences, val_inference = None, None
        test_sequences, test_inference = load_remote_homology("test_fold_holdout")
    if task == "remote_homology_test_family_holdout":
        train_sequences, train_inference = None, None
        val_sequences, val_inference = None, None
        test_sequences, test_inference = load_remote_homology("test_family_holdout")
    if task == "remote_homology_test_superfamily_holdout":
        train_sequences, train_inference = None, None
        val_sequences, val_inference = None, None
        test_sequences, test_inference = load_remote_homology("test_superfamily_holdout")

    if task == "fluorescence":
        train_sequences, train_inference = load_fluorescence("train")
        val_sequences, val_inference = load_fluorescence("valid")
        test_sequences, test_inference = load_fluorescence("test")

    if task == "stability":
        train_sequences, train_inference = load_stability("train")
        val_sequences, val_inference = load_stability("valid")
        test_sequences, test_inference = load_stability("test")
        
    return train_sequences, train_inference, val_sequences, val_inference, test_sequences, test_inference


def create_normalizer(train_list_of_lists):
    """
    Create a normalizer function based on the training list of lists.

    :param train_list_of_lists: The training list of lists used to determine the normalization parameters
    :return: A function that can be used to normalize other list of lists with the same parameters
    """
    # Flatten the training list of lists and convert to NumPy array
    flattened_train = np.hstack(train_list_of_lists)

    # Determine the normalization parameters (min and max)
    min_val = np.min(flattened_train)
    max_val = np.max(flattened_train)

    def normalizer(list_of_lists):
        """
        Normalize a list of lists using the predetermined min and max values.

        :param list_of_lists: A list containing lists of numerical values
        :return: A list of lists with normalized values
        """
        flattened = np.hstack(list_of_lists)
        norm_flattened = (flattened - min_val) / (max_val - min_val)

        # Reshape back into the original list of lists structure
        normalized_list_of_lists = []
        start = 0
        for sublist in list_of_lists:
            end = start + len(sublist)
            normalized_list_of_lists.append(norm_flattened[start:end].tolist())
            start = end

        return normalized_list_of_lists

    return normalizer


###### Create Samples ######
def format_sequence_and_label(sequence, label, sp, banned_tokens):
    """
    Tokenizes a sequence using SentencePiece and formats the sequence with corresponding label modes.

    Args:
        sequence (str): The input sequence to be tokenized.
        label (List[int]): The label values corresponding to each token in the sequence.
        sp: An instance of SentencePiece model for tokenization.

    Returns:
        Tuple[List[int], List[int], List[int], int]: A tuple containing tokenized sequence,
        label modes, token lengths, and sequence length.
    """
    tokenized_sequence = sp.EncodeAsIds(sequence)
    tokenized_sequence = [x for x in tokenized_sequence if x not in banned_tokens]
    
    token_lens = [len(sp.DecodeIds(token)) for token in tokenized_sequence]
    
    label_modes = []
    ptr = 0
    for i, length in enumerate(token_lens):
        if tokenized_sequence[i] == 0:  # this is the unknown token
            length = 1
        try:
            label_modes.append(np.bincount(label[ptr:ptr+length]).argmax())
            
        except:
            print(label[ptr:ptr+length])
            print(label)
            print(len(label))
            print(ptr, length)
            print(tokenized_sequence)
            print(token_lens)
            print(sequence)
            raise
        
        ptr += length
    sequence_length = len(sequence)
    return tokenized_sequence, label_modes, token_lens, sequence_length

def format_sequence_and_single_label(sequence, label, sp, banned_tokens):
    """
    Tokenizes a sequence using SentencePiece and formats the sequence with corresponding label modes.

    Args:
        sequence (str): The input sequence to be tokenized.
        label (List[int]): The label values corresponding to each token in the sequence.
        sp: An instance of SentencePiece model for tokenization.

    Returns:
        Tuple[List[int], List[int], List[int], int]: A tuple containing tokenized sequence,
        label modes, token lengths, and sequence length.
    """
    tokenized_sequence = sp.EncodeAsIds(sequence)
    tokenized_sequence = [x for x in tokenized_sequence if x not in banned_tokens]
    
    token_lens = [len(sp.DecodeIds(token)) for token in tokenized_sequence]
    
    sequence_length = len(sequence)

    return tokenized_sequence, label, token_lens, sequence_length

def format_sequence_and_value(sequence, label, sp, banned_tokens):
    """
    Tokenizes a sequence using SentencePiece and formats the sequence with corresponding label value means.

    Args:
        sequence (str): The input sequence to be tokenized.
        label (List[int]): The label values corresponding to each token in the sequence.
        sp: An instance of SentencePiece model for tokenization.

    Returns:
        Tuple[List[int], List[float], List[int], int]: A tuple containing tokenized sequence,
        label value means, token lengths, and sequence length.
    """
    tokenized_sequence = sp.EncodeAsIds(sequence)
    tokenized_sequence = [x for x in tokenized_sequence if x not in banned_tokens]
    
    token_lens = [len(sp.DecodeIds(token)) for token in tokenized_sequence]
    
    label_means = []
    ptr = 0
    for i, length in enumerate(token_lens):
        if tokenized_sequence[i] == 0:  # this is the unknown token
            length = 1
        label_means.append(np.mean(label[ptr:ptr+length]))
        ptr += length
    sequence_length = len(sequence)
    return tokenized_sequence, label_means, token_lens, sequence_length

def format_sequence_and_single_value(sequence, label, sp, banned_tokens):
    """
    Tokenizes a sequence using SentencePiece and formats the sequence with corresponding label value means.

    Args:
        sequence (str): The input sequence to be tokenized.
        label (int): The label values corresponding to each token in the sequence.
        sp: An instance of SentencePiece model for tokenization.

    Returns:
        Tuple[List[int], List[float], List[int], int]: A tuple containing tokenized sequence,
        label value means, token lengths, and sequence length.
    """
    tokenized_sequence = sp.EncodeAsIds(sequence)
    tokenized_sequence = [x for x in tokenized_sequence if x not in banned_tokens]
    
    token_lens = [len(sp.DecodeIds(token)) for token in tokenized_sequence]
    sequence_length = len(sequence)

    return tokenized_sequence, label, token_lens, sequence_length


def process_data(sp, sequences, target, format_func, prefix, banned_tokens=[]):
    """
    Tokenizes the sequences, prepares targets, generates embeddings, and flattens the data.

    Args:
        sequences (List[str]): List of input sequences.
        ss3 (List[List[int]]): List of label values corresponding to each token in the sequences.
        format_func (Callable): Function for formatting sequences and labels (either format_sequence_and_label or format_sequence_and_value).
        model: The model used for generating embeddings.
        sp: An instance of SentencePiece model for tokenization.
        device: The device on which the model and tensors are placed.

    Returns:
        Tuple[np.ndarray, np.ndarray, List[np.ndarray]]: A tuple containing flattened embeddings,
        flattened targets, and the list of embeddings for each sequence.
    """
    if sequences is None:
        return None, None
    
    # Tokenizing the sequences and preparing the targets
    sequences_tokenized = []
    targets = []

    for i in tqdm(range(len(sequences))):
        tokenized_sequence, target_values, _, _ = format_func(sequences[i], target[i], sp, banned_tokens)

        sequences_tokenized.append(tokenized_sequence)
        targets.append(target_values)

    # Encoding sequences
    sequences_tokenized = [prefix + item + [EOS_TOKEN] for item in sequences_tokenized]
    return sequences_tokenized, targets

def get_training_sets(task, sp, format_func, prefix, banned_tokens):
    X_train_raw, y_train_raw, X_val_raw, y_val_raw, X_test_raw, y_test_raw = load_task(task)
    X_train, y_train = process_data(sp, X_train_raw, y_train_raw, format_func, prefix, banned_tokens)
    X_val, y_val = process_data(sp, X_val_raw, y_val_raw, format_func, prefix, banned_tokens)
    X_test, y_test = process_data(sp, X_test_raw, y_test_raw, format_func, prefix, banned_tokens)

    if task in ["fluorescence", "stability"]:
        mean = np.mean(y_train)
        std = np.std(y_train)
        y_train = [(y - mean) / std for y in y_train]
        y_val = [(y - mean) / std for y in y_val]
        y_test = [(y - mean) / std for y in y_test]
    
    return X_train, y_train, X_val, y_val, X_test, y_test


###### Finetune Models ######
@torch.jit.script
def block_attn(attn_mask: torch.Tensor, start: int, end: int, batch_idx: int) -> int:
    # create zeros in all places where attention should happen
    attn_mask[batch_idx, start:end, start:end] = 0
    return 0

@torch.jit.script
def create_attention_mask(attn_mask: torch.Tensor, input_ids: torch.Tensor, EOS_TOKEN: int = 3, padding: bool = False) -> torch.Tensor:
    if not padding:
        temp = torch.ones(input_ids.size(0), input_ids.size(1) + 1, device=input_ids.device, dtype=input_ids.dtype)
        temp[:, :-1] = input_ids
        temp[:, -1] = EOS_TOKEN
        input_ids = temp

    EOS_positions = (input_ids == EOS_TOKEN).nonzero()
    attn_mask.fill_(-1e9)

    prev_index = 0
    prev_batch_idx = 0
    for i in range(0, len(EOS_positions)):
        if EOS_positions[i][0] == prev_batch_idx: # make sure the EOS positions are in the batch
            block_attn(attn_mask, prev_index, EOS_positions[i][1] + 1, prev_batch_idx)
            prev_index = EOS_positions[i][1] + 1
        else:
            prev_batch_idx = EOS_positions[i][0]
            prev_index = 0
            block_attn(attn_mask, prev_index, EOS_positions[i][1] + 1, prev_batch_idx)

    for i in range(0, len(input_ids)):
        if not torch.any(EOS_positions[:, 0] == i):
            attn_mask[i, :, :] = 0
    
    return attn_mask

def pad_attn(attn_mask, x):
    pad_locations = (x == PAD_TOKEN).nonzero()
    for i in range(0, len(pad_locations)):
        attn_mask[pad_locations[i][0], pad_locations[i][1] + 1:, :] = -1e9
        attn_mask[pad_locations[i][0], :, pad_locations[i][1] + 1:] = -1e9
    
    return attn_mask


def finetune_on_task(task, model, sp, banned_tokens, format_func, metric, 
                     loss_str, device, dtype=torch.bfloat16, num_epochs=4, batch_size=1, 
                     num_accumulation_steps=8, lr=1e-4, embed_lr=1e-2, test_freq=100, single_target=False):
    # Load the task
    prefix = [tok for tok in sp.EncodeAsIds("<protein>") if tok not in banned_tokens]

    X_train, Y_train, X_val, Y_val, _, _ = get_training_sets(task, sp, format_func, prefix, banned_tokens)

    base_model = copy.deepcopy(model)
    base_model.train()

    if single_target:
        output_dim = 1 if loss_str == 'mse' else max(Y_train)+1
    else:
        output_dim = 1 if loss_str == 'mse' else max([item for sublist in Y_train for item in sublist])+1
    head = torch.nn.Linear(base_model.transformer.wte.weight.shape[-1], output_dim).to(device).to(dtype)

    param_groups = [
        {"params": [p for name, p in base_model.named_parameters() if "wte" in name], "lr": embed_lr},
        {"params": [p for name, p in base_model.named_parameters() if "wte" not in name], "lr": lr},
        {"params": head.parameters(), "lr": 1e-2}
    ]
    
    num_steps = int(num_epochs*len(X_train)/(batch_size*num_accumulation_steps))
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=num_steps)
    if loss_str == 'mse':
        loss_fn = torch.nn.MSELoss()
    elif loss_str == 'cross_entropy':
        loss_fn = torch.nn.CrossEntropyLoss()
    else:
        raise ValueError("Unknown loss function")

    if metric == "SCC":
        metric_fn = lambda y,pred : spearmanr(y,pred)[0]
    if metric == "ACC":
        metric_fn = accuracy_score

    loss_hist = []
    val_pccs = []
    last_val_loss = 0
    best_val_pcc = -2 # for spearmanr, the minimum is -1, so this will always be overwritten
    pbar = tqdm(range(num_steps))
    for step in pbar:
        optimizer.zero_grad()
        total_loss = 0.0
        model.train()
        head.train()
        for _ in range(num_accumulation_steps):
            indices = np.random.choice(len(X_train), batch_size, replace=False)
            
            X = np.array(X_train, dtype=object)[indices]
            
            X = np.array([x[:1024] for x in X], dtype=object)
            target = np.array(Y_train, dtype=object)[indices]
            if not single_target:
                target = np.array([tar[:1023] for tar in target], dtype=object)
            
            for x, Y in zip(X, target):
                x = torch.tensor(x, device=device, dtype=torch.long).unsqueeze(0)
                embeddings = model(x, return_embeddings=True)

                if single_target:
                    embeddings = embeddings[:, 0]
                else:
                    embeddings = embeddings[:, 1:len(Y)+1]
                
                output = head(embeddings).squeeze(0)
                if loss_str == 'mse':
                    if single_target:
                        y = torch.tensor(Y, device=device, dtype=dtype).unsqueeze(0)
                    else:
                        y = torch.tensor(Y, device=device, dtype=dtype).unsqueeze(1)
                else:
                    y = torch.tensor(Y, device=device, dtype=torch.long)
                    
                loss = loss_fn(output, y) / (num_accumulation_steps*batch_size)
                loss.backward()

                total_loss += loss.item()
        
        loss_hist.append(total_loss)

        optimizer.step()
        scheduler.step()

        pbar.set_description(f"Loss: {total_loss:.4f}, Val {metric}: {last_val_loss:.4f}")

        if step % max(1, num_steps // test_freq) == 0:
            model.eval()
            head.eval()
            with torch.no_grad():
                ground_truths = []
                predictions = []
                for i in range(0, len(X_val)):
            
                    x = np.array(X_val, dtype=object)[i]
                    
                    x = x[:1024]
                    y = np.array(Y_val, dtype=object)[i]
                    if not single_target:
                        y = y[:1023]
                    
                    x = torch.tensor(x, device=device, dtype=torch.long).unsqueeze(0)
                    embeddings = model(x, return_embeddings=True)

                    if single_target:
                        embeddings = embeddings[:, 0]
                    else:
                        embeddings = embeddings[:, 1:len(y)+1]

                    output = head(embeddings).squeeze(0)
                    y = torch.tensor(y, device=device, dtype=dtype)
                    
                    if single_target:
                        ground_truths.append(y.cpu().item())
                    else:
                        ground_truths.extend(y.cpu().tolist())

                    if loss_str == 'mse':
                        predictions.extend(output.reshape(-1).cpu().tolist())
                    else:
                        if single_target:
                            predictions.append(output.argmax(dim=-1).cpu().item())
                        else:
                            predictions.extend(output.argmax(dim=-1).cpu().numpy().tolist())

                    pbar.set_description(f"(Testing {i}/{len(X_val)}) Loss: {total_loss:.4f}, Val metric: {last_val_loss:.4f}")

                ground_truths = np.asarray(ground_truths)
                predictions = np.asarray(predictions)
                last_val_loss = metric_fn(ground_truths, predictions)
                val_pccs.append(last_val_loss)
                
                if last_val_loss > best_val_pcc:
                    best_val_pcc = last_val_loss
                    best_model_sd = copy.deepcopy(model.state_dict())
                    best_head_sd = copy.deepcopy(head.state_dict())

    base_model.load_state_dict(best_model_sd)
    head.load_state_dict(best_head_sd)
    return base_model, head
    

def test_model(task, model, head, sp, banned_tokens, format_func, metric, loss_str, device, dtype=torch.bfloat16, single_target=False):
    # Test the model
    model.eval()
    head.eval()
    prefix = [tok for tok in sp.EncodeAsIds("<protein>") if tok not in banned_tokens]
    _, _, _, _, X_test, Y_test = get_training_sets(task, sp, format_func, prefix, banned_tokens)

    if metric == "SCC":
        metric_fn = lambda y,pred : spearmanr(y,pred)[0]
    if metric == "ACC":
        metric_fn = accuracy_score

    with torch.no_grad():
        predictions = []
        ground_truths = []
        for i in range(0, len(X_test)):
            
            x = np.array(X_test, dtype=object)[i]
            
            x = x[:1024]
            y = np.array(Y_test, dtype=object)[i]
            if not single_target:
                y = y[:1023]
            x = torch.tensor(x, device=device, dtype=torch.long).unsqueeze(0)
            embeddings = model(x, return_embeddings=True)
            if not single_target:
                embeddings = embeddings[:, 1:len(y)+1]
            else:
                embeddings = embeddings[:, 0]

            output = head(embeddings).squeeze(0)
            y = torch.tensor(y, device=device, dtype=dtype)
            
            
            if single_target:
                ground_truths.append(y.cpu().item())
            else:
                ground_truths.extend(y.cpu().tolist())

            if loss_str == 'mse':
                predictions.extend(output.reshape(-1).cpu().tolist())
            else:
                if single_target:
                    predictions.append(output.argmax(dim=-1).cpu().item())
                else:
                    predictions.extend(output.argmax(dim=-1).cpu().numpy().tolist())


        ground_truths = np.asarray(ground_truths)
        predictions = np.asarray(predictions)
        performance = metric_fn(ground_truths, predictions)

        print(f"{metric}: {performance}")
        return performance

def main(sp_dir, model_dir, banned_token=[65533], extra_pretrain=False, pretraining_epochs=4, pretraining_num_accum_steps=4, batch_size=32, pretraining_lr=1e-3, finetuning_lr=2e-4, output_suffix=""):
    print(f"Loading tokenizer from {sp_dir}...")
    print(f"Loading model from {model_dir}...")
    print(f"Using banned token {banned_token}")
    print(f"Pretraining for {pretraining_epochs} epochs with {pretraining_num_accum_steps} accumulation steps, batch size {batch_size}, lr {pretraining_lr}")
    print(f"Finetuning with lr {finetuning_lr}")
    print(f"Saving with output suffix: {output_suffix}")

    tasks = ["structure_ss3", "structure_ss8", "remote_homology", "fluorescence", "stability"]
    dtype = torch.bfloat16

    sp = spm.SentencePieceProcessor()
    sp.Load(sp_dir)
    if not isinstance(banned_token, list):
        banned_tokens = [banned_token]
    else:
        banned_tokens = banned_token

    model = torch.load(model_dir, map_location=device).to(dtype).to(device)
    model.eval()

    print(f"Num params: {model.get_num_params() / 10**6:.2f}M")

    ########################### finetune on tasks ###########################
    results_ft = {}
    for task in tasks:
        if "ss3" in task or "ss8" in task:
            metric = 'ACC'
            loss = "cross_entropy"
            format_func = format_sequence_and_label
        elif "remote_homology" in task:
            metric = 'ACC'
            loss = "cross_entropy"
            format_func = format_sequence_and_single_label
        elif "fluorescence" in task or "stability" in task:
            metric = 'SCC'
            loss = "mse"
            format_func = format_sequence_and_single_value
        else:
            raise ValueError("Unknown task")
        
        batch_size = 32
        epochs = 64
        num_accumulation_steps = 1
        embed_lr = 1e-3
        print("---------------------------------------------------------------")
        print(f"Evaluting task {task}, training for {epochs} epochs...")
        base_model, head = finetune_on_task(task, model, sp, banned_tokens, format_func, metric, 
                                            loss, device, dtype=dtype, batch_size=batch_size, 
                                            num_accumulation_steps=num_accumulation_steps, num_epochs=epochs, 
                                            lr=finetuning_lr, embed_lr=embed_lr, test_freq=100, 
                                            single_target="fluorescence" in task or "stability" in task or "remote_homology" in task)
        
        if "ss3" not in task and "ss8" not in task and "remote_homology" not in task:
            result = test_model(task, base_model, head, sp, banned_tokens, format_func, metric, loss, device, dtype=dtype, single_target=True)
            results_ft[task] = result

            print("---------------------------------------------------------------")
            print(f'{result} for task {task}')
        
        if "ss3" in task:
            for subtest in ["casp12_ss3", "cb513_ss3", "ts115_ss3"]:
                result = test_model(subtest, base_model, head, sp, banned_tokens, format_func, metric, loss, device, dtype=dtype)
                print("---------------------------------------------------------------")
                print(f'{result} for task {subtest}')
                results_ft[subtest] = result
        if "ss8" in task:
            for subtest in ["casp12_ss8", "cb513_ss8", "ts115_ss8"]:
                result = test_model(subtest, base_model, head, sp, banned_tokens, format_func, metric, loss, device, dtype=dtype)
                print("---------------------------------------------------------------")
                print(f'{result} for task {subtest}')
                results_ft[subtest] = result
        if "remote_homology" in task:
            for subtest in ["remote_homology_test_fold_holdout", "remote_homology_test_family_holdout", "remote_homology_test_superfamily_holdout"]:
                result = test_model(subtest, base_model, head, sp, banned_tokens, format_func, metric, loss, device, dtype=dtype, single_target=True)
                print("---------------------------------------------------------------")
                print(f'{result} for task {subtest}')
                results_ft[subtest] = result
        
    with open(f"TAPE_{output_suffix}_results.csv", "w") as f:
        for task, result in results_ft.items():
            f.write(f"{task},{result}\n")

if __name__ == "__main__":
    #main('/home/sully/OmniBioTA/tokenizers/bpe/mixed_bpe.model', '/home/sully/OmniBioTA/models/omnibiota-multi-E-mixed-small.pt', [65533], finetuning_lr=2e-4, extra_pretrain=False)
    fire.Fire(main)