import os
import sys
sys.path.insert(0, '../training')
from loader import EOS_TOKEN, PAD_TOKEN, MASK_TOKEN
import numpy as np
import torch
import sentencepiece as spm
from tqdm import tqdm
import copy
from sklearn.metrics import matthews_corrcoef, f1_score
from sklearn.metrics import roc_auc_score, accuracy_score
from scipy.stats import pearsonr

import pickle
import pandas as pd
import re
import os
import fire

device = "cuda:0" if torch.cuda.is_available() else "cpu"


###### Clean Raw CSVs ######
def get_cleaned_evals(eval_dir, force = False):
    dfs = [] # will store the dataframes for each CSV
    fns = [] # will store the filenames of each CSV

    # TODO: Add error handling for the file not existsing
    # Check if the directory exists
    if not os.path.exists(eval_dir):
        print(f"Directory {eval_dir} does not exist.")
        return
    # check if file exists
    if os.path.isfile("../datasets/ProteinGLUE_processed.pkl") or not force:
        with open("../datasets/ProteinGLUE_processed.pkl", "rb") as f:
            datasets = pickle.load(f)
        return datasets
    
    for fn in os.listdir(eval_dir):
        if 'csv' in fn: # only read CSV files
            df = pd.read_csv(eval_dir + "/" + fn)

            # remove the brackets, quotes, and newlines from the labels
            for col in df.columns:
                df[col] = df[col].apply(lambda x: re.sub("[\[\]\'b\n]", '', x))
            dfs.append(df)
            fns.append(fn)

    names = [fn[:-4] for fn in fns] # remove the .csv from the filename to get the dataset name

    datasets = {} # will store the datasets
    for name, df in zip(names, dfs):
        datasets[name] = {} # initialize the dataset

        datasets[name]['sequences'] = [] # will store the sequences
        label_columns = [item for item in df.columns.tolist() if item != 'sequence'] # get the label columns, excluding the sequence column
        error_indices = [] # will store the indices of the labels with errors so we don't append the corresponding sequence to the dataset

        for label_column in label_columns: # iterate through the label columns
            raw_labels = df[label_column].tolist() # get the raw labels

            num_errors = 0 # will store the number of labels with errors
            labels = [] # will store the labels
            for i, raw_label in enumerate(raw_labels):
                if "..." in raw_label.split(): # some labels have "..." in them for some reason
                    num_errors += 1
                    error_indices.append(i) # add the index of the label with an error
                    continue # skip this label

                labels.append([float(item) for item in raw_label.split()]) # convert the label to a list of floats and append it to the labels list
            
            datasets[name][label_column] = labels # add the labels to the dataset
            print("Dataset: {}, Label: {}, Num Errors: {}".format(name, label_column, num_errors)) # print the number of errors for this label
        
        # add the sequences to the dataset
        raw_sequences = df['sequence'].to_list() # get the raw sequences
        for i in range(0, len(raw_sequences)):
            if i not in error_indices: # if the label at this index doesn't have an error
                datasets[name]['sequences'].append(raw_sequences[i])

        # verify the dataset
        for label in datasets[name].keys():
            # check that the number of labels is the same as the number of sequences
            if len(datasets[name][label]) != len(datasets[name]['sequences']):
                print(f"ERROR {name} {label} {len(datasets[name][label])} {len(datasets[name]['sequences'])}")
                continue

            # check that the length of each label is the same as the length of the corresponding sequence
            for i in range(0, len(datasets[name][label])):
                if len(datasets[name][label][i]) != len(datasets[name]["sequences"][i]):
                    print(f"ERROR {name} {label} {i} {len(datasets[name][label][i])} {len(datasets[name]['sequences'][i])}")
            print()
    return datasets


def load_task(task, datasets):
    if "SS3" == task:
        train_sequences = datasets["ss_training"]['sequences']
        train_inference = datasets["ss_training"]['ss3']
        val_sequences = datasets["ss_validation"]['sequences']
        val_inference = datasets["ss_validation"]['ss3'] 
        test_sequences = datasets["ss_test"]['sequences']
        test_inference = datasets["ss_test"]['ss3']
    elif "SS8" == task:
        train_sequences = datasets["ss_training"]['sequences']
        train_inference = datasets["ss_training"]['ss8']
        val_sequences = datasets["ss_validation"]['sequences']
        val_inference = datasets["ss_validation"]['ss8'] 
        test_sequences = datasets["ss_test"]['sequences']
        test_inference = datasets["ss_test"]['ss8']
    elif "CB513SS8" == task:
        train_sequences = None
        train_inference = None
        val_sequences = None
        val_inference = None
        test_sequences = datasets["ss_cb513_test"]['sequences']
        test_inference = datasets["ss_cb513_test"]['ss8']
    elif "CB513SS3" == task:
        train_sequences = None
        train_inference = None
        val_sequences = None
        val_inference = None
        test_sequences = datasets["ss_cb513_test"]['sequences']
        test_inference = datasets["ss_cb513_test"]['ss3']
    elif "BUR" == task:
        train_sequences = datasets["asabu_training"]['sequences']
        train_inference = datasets["asabu_training"]['buried']
        val_sequences = datasets["asabu_validation"]['sequences']
        val_inference = datasets["asabu_validation"]['buried'] 
        test_sequences = datasets["asabu_test"]['sequences']
        test_inference = datasets["asabu_test"]['buried']
    elif "ASA" == task:
        train_sequences = datasets["asabu_training"]['sequences']
        train_inference = datasets["asabu_training"]['solvent_accessibility']
        val_sequences = datasets["asabu_validation"]['sequences']
        val_inference = datasets["asabu_validation"]['solvent_accessibility'] 
        test_sequences = datasets["asabu_test"]['sequences']
        test_inference = datasets["asabu_test"]['solvent_accessibility']
    elif "PPI" == task:
        train_sequences = datasets["ppi_hetro_homo_training"]['sequences']
        train_inference = datasets["ppi_hetro_homo_training"]['interface']
        val_sequences = datasets["ppi_hetro_homo_validation"]['sequences']
        val_inference = datasets["ppi_hetro_homo_validation"]['interface'] 
        test_sequences = datasets["ppi_hetro_homo_test"]['sequences']
        test_inference = datasets["ppi_hetro_homo_test"]['interface']
    elif "EPI" == task:
        train_sequences = datasets["Epitope_anti_training_1"]['sequences']
        train_inference = datasets["Epitope_anti_training_1"]['interface']
        val_sequences = datasets["Epitope_anti_validation_1"]['sequences']
        val_inference = datasets["Epitope_anti_validation_1"]['interface'] 
        test_sequences = datasets["Epitope_anti_test_1"]['sequences']
        test_inference = datasets["Epitope_anti_test_1"]['interface']
    elif "HPR" == task:
        train_sequences = datasets["HPrank_training"]['sequences']
        train_inference = datasets["HPrank_training"]['hydrophobic_patch']
        val_sequences = datasets["HPrank_validation"]['sequences']
        val_inference = datasets["HPrank_validation"]['hydrophobic_patch'] 
        test_sequences = datasets["HPrank_test"]['sequences']
        test_inference = datasets["HPrank_test"]['hydrophobic_patch']
        
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

def get_training_sets(task, dataset, sp, format_func, prefix, banned_tokens):
    X_train_raw, y_train_raw, X_val_raw, y_val_raw, X_test_raw, y_test_raw = load_task(task, dataset)
    if task not in ["CB513SS3", "CB513SS8"]:
        X_train, y_train = process_data(sp, X_train_raw, y_train_raw, format_func, prefix, banned_tokens)
        X_val, y_val = process_data(sp, X_val_raw, y_val_raw, format_func, prefix, banned_tokens)
    else:
        X_train, y_train = None, None
        X_val, y_val = None, None
    X_test, y_test = process_data(sp, X_test_raw, y_test_raw, format_func, prefix, banned_tokens)

    if task in ["ASA", "HPR"]:
        normalizer = create_normalizer(y_train_raw)
        y_train = normalizer(y_train)
        y_val = normalizer(y_val)
        y_test = normalizer(y_test)
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


def finetune_on_task(task, model, dataset, sp, banned_tokens, format_func, metric, loss_str, device, dtype=torch.bfloat16, num_epochs=4, batch_size=1, num_accumulation_steps=8, lr=1e-4, embed_lr=1e-2, test_freq=100):
    # Load the task
    prefix = [tok for tok in sp.EncodeAsIds("<protein>") if tok not in banned_tokens]

    X_train, Y_train, X_val, Y_val, _, _ = get_training_sets(task, dataset, sp, format_func, prefix, banned_tokens)

    base_model = copy.deepcopy(model)
    base_model.train()

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

    if metric == "PCC":
        metric_fn = lambda y,pred : pearsonr(y,pred)[0]
    if metric == "ACC":
        metric_fn = accuracy_score
    if metric == "AUC":
        metric_fn = roc_auc_score

    

    loss_hist = []
    val_pccs = []
    last_val_loss = 0
    best_val_pcc = 0
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
            target = np.array([tar[:1023] for tar in target], dtype=object)
            for x, Y in zip(X, target):
                x = torch.tensor(x, device=device, dtype=torch.long).unsqueeze(0)
                embeddings = model(x, return_embeddings=True)
                embeddings = embeddings[:, 1:len(Y)+1]
                output = head(embeddings).squeeze(0)
                if loss_str == 'mse':
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
                    y = y[:1023]
                    x = torch.tensor(x, device=device, dtype=torch.long).unsqueeze(0)
                    embeddings = model(x, return_embeddings=True)
                    embeddings = embeddings[:, 1:len(y)+1]
                    output = head(embeddings).squeeze(0)
                    y = torch.tensor(y, device=device, dtype=dtype)
                    

                    ground_truths.extend(y.cpu().tolist())
                    if loss_str == 'mse':
                        predictions.extend(output.reshape(-1).cpu().tolist())
                    else:
                        predictions.extend(output.argmax(dim=-1).cpu().numpy().tolist())

                    pbar.set_description(f"(Testing {i}/{len(X_val)}) Loss: {total_loss:.4f}, Val pcc: {last_val_loss:.4f}")

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
    

def test_model(task, model, head, dataset, sp, banned_tokens, format_func, metric, loss_str, device, dtype=torch.bfloat16):
    # Test the model
    model.eval()
    head.eval()
    prefix = [tok for tok in sp.EncodeAsIds("<protein>") if tok not in banned_tokens]
    _, _, _, _, X_test, Y_test = get_training_sets(task, dataset, sp, format_func, prefix, banned_tokens)

    if metric == "PCC":
        metric_fn = lambda y,pred : pearsonr(y,pred)[0]
    if metric == "ACC":
        metric_fn = accuracy_score
    if metric == "AUC":
        metric_fn = roc_auc_score

    with torch.no_grad():
        predictions = []
        ground_truths = []
        for i in range(0, len(X_test)):
            
            x = np.array(X_test, dtype=object)[i]
            
            x = x[:1024]
            y = np.array(Y_test, dtype=object)[i]
            y = y[:1023]
            x = torch.tensor(x, device=device, dtype=torch.long).unsqueeze(0)
            embeddings = model(x, return_embeddings=True)
            embeddings = embeddings[:, 1:len(y)+1]
            output = head(embeddings).squeeze(0)
            y = torch.tensor(y, device=device, dtype=dtype)
            
            
            ground_truths.extend(y.cpu().tolist())
            if loss_str == 'mse':
                predictions.extend(output.reshape(-1).cpu().tolist())
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

    tasks = ["SS3", "SS8","ASA", "HPR", "PPI", "BUR", "EPI"]
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

    dataset = get_cleaned_evals('../datasets/ProteinGLUE', force = False)

    ########################### pre-train on sequence data ###########################
    # TODO fix the sequence loader
    if extra_pretrain:
        all_sequences = []
        for task in tasks:
            prefix = [tok for tok in sp.EncodeAsIds("<protein>") if tok not in banned_tokens]
            X_train = get_training_sets(task, dataset, sp, format_sequence_and_label, prefix, banned_tokens)[0]
            all_sequences += X_train
        loss_hist = []
        num_epochs = pretraining_epochs
        num_accumulation_steps = pretraining_num_accum_steps
        lr = pretraining_lr
        embed_lr = 1e-2

        num_steps = int(num_epochs*len(all_sequences)/(batch_size*num_accumulation_steps))
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=num_steps)
        pbar = tqdm(range(num_steps))

        for step in pbar:
            optimizer.zero_grad()
            total_loss = 0
            for _ in range(num_accumulation_steps):
                indices = np.random.choice(len(all_sequences), batch_size, replace=False)
                x = torch.ones((batch_size, max([len(all_sequences[i]) for i in indices])), dtype=torch.long, device=device) * PAD_TOKEN
                for i, idx in enumerate(indices):
                    x[i][:len(all_sequences[idx])] = torch.tensor(all_sequences[idx], dtype=torch.long, device=device)
                
                #attn_mask = create_attention_mask(x, model.transformer.h[0].attn.n_head).to(dtype).to(device)
                attn_mask = torch.zeros((x.shape[0], x.shape[-1], x.shape[-1])).to(device)
                attn_mask = pad_attn(attn_mask, x)
                attn_mask = attn_mask.unsqueeze(1).expand(-1, model.transformer.h[0].attn.n_head, -1, -1).to(dtype)
                token_mask = torch.rand(x.shape) < 0.15
                masked_tokens = x.clone()
                masked_tokens[token_mask] = MASK_TOKEN

                out = model(masked_tokens, attn_mask=attn_mask)
                loss = torch.nn.functional.cross_entropy(out.view(-1, out.shape[-1]), x.view(-1), ignore_index=PAD_TOKEN, reduction="sum") / (x != PAD_TOKEN).sum() / num_accumulation_steps
                
                loss.backward()
                total_loss += loss.item()
            
            optimizer.step()
            scheduler.step()

            pbar.set_description(f"Loss: {total_loss:.4f}")
            loss_hist.append(total_loss)
    
    ########################### finetune on tasks ###########################
    results_ft = {}
    for task in tasks:
        if "SS3" in task:
            metric = 'ACC'
            loss = "cross_entropy"
            format_func = format_sequence_and_label
        elif "SS8" in task:
            metric = 'ACC'
            loss = "cross_entropy"
            format_func = format_sequence_and_label
        elif "BUR" in task:
            metric = 'ACC'
            loss = "cross_entropy"
            format_func = format_sequence_and_label
        elif "ASA" in task:
            metric = 'PCC'
            loss = "mse"
            format_func = format_sequence_and_value
        elif "PPI" in task:
            metric = 'AUC'
            loss = "cross_entropy"
            format_func = format_sequence_and_label
        elif "EPI" in task:
            metric = 'AUC'
            loss = "cross_entropy"
            format_func = format_sequence_and_label
        elif "HPR" in task:
            metric = 'PCC'
            loss = "mse"
            format_func = format_sequence_and_value
        else:
            raise ValueError("Unknown task")
        
        batch_size = 32
        epochs = 16 if "EPI" in task or "PPI" in task else 64
        num_accumulation_steps = 1
        embed_lr = 1e-2
        print("---------------------------------------------------------------")
        print(f"Evaluting task {task}, training for {epochs} epochs...")
        base_model, head = finetune_on_task(task, model, dataset, sp, banned_tokens, format_func, metric, loss, device, dtype=dtype, batch_size=batch_size, num_accumulation_steps=num_accumulation_steps, num_epochs=epochs, lr=finetuning_lr, embed_lr=embed_lr, test_freq=100)
        
        result = test_model(task, base_model, head, dataset, sp, banned_tokens, format_func, metric, loss, device, dtype=dtype)
        results_ft[task] = result
        
        print("---------------------------------------------------------------")
        print(f'{result} for task {task}')
        if task == "SS3":
            result = test_model('CB513SS3', base_model, head, dataset, sp, banned_tokens, format_func, metric, loss, device, dtype=dtype)
            print("---------------------------------------------------------------")
            print(f'{result} for task CB513SS3')
            results_ft["CB513SS3"] = result
        if task == "SS8":
            result = test_model('CB513SS8', base_model, head, dataset, sp, banned_tokens, format_func, metric, loss, device, dtype=dtype)
            print("---------------------------------------------------------------")
            print(f'{result} for task CB513SS8')
            results_ft["CB513SS8"] = result
    
    with open(f"ProteinGLUE_{output_suffix}_results.csv", "w") as f:
        for task, result in results_ft.items():
            f.write(f"{task},{result}\n")

if __name__ == "__main__":
    #main(r'c:\Users\sully\Desktop\tokenizers\bpe\mixed_bpe.model', r'c:\Users\sully\Desktop\omnibiota-multi-E-mixed-medium_1024ctx_250b.pt', [65533], finetuning_lr=2e-4, extra_pretrain=False)
    fire.Fire(main)