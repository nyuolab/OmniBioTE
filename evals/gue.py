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
import fire

device = "cuda:0"

def pad_attn(attn_mask, x):
    pad_locations = (x == PAD_TOKEN).nonzero()
    for i in range(0, len(pad_locations)):
        attn_mask[pad_locations[i][0], pad_locations[i][1] + 1:, :] = -1e9
        attn_mask[pad_locations[i][0], :, pad_locations[i][1] + 1:] = -1e9
    
    return attn_mask

def load_task(task_dir):
    with open(os.path.join(task_dir, "train.csv"), "r") as f:
        X_train = []
        Y_train = []
        lines = f.readlines()
        for line in lines[1:]:
            X_train.append(line.split(",")[0])
            Y_train.append(line.split(",")[1])

    with open(os.path.join(task_dir, "dev.csv"), "r") as f:
        X_val = []
        Y_val = []
        lines = f.readlines()
        for line in lines[1:]:
            X_val.append(line.split(",")[0])
            Y_val.append(line.split(",")[1])

    with open(os.path.join(task_dir, "test.csv"), "r") as f:
        X_test = []
        Y_test = []
        lines = f.readlines()
        for line in lines[1:]:
            X_test.append(line.split(",")[0])
            Y_test.append(line.split(",")[1])

    return X_train, Y_train, X_val, Y_val, X_test, Y_test

def finetune_on_task(task, model, sp, banned_tokens, device, dtype=torch.bfloat16, num_epochs=4, batch_size=4, num_accumulation_steps=8, lr=1e-4, embed_lr=1e-2, test_freq=100):
    base_model = copy.deepcopy(model)
    base_model.train()

    X_train, Y_train, X_val, Y_val, X_test, Y_test = load_task(task)

    Y_train = np.array([int(y) for y in Y_train])
    Y_val = np.array([int(y) for y in Y_val])
    Y_test = np.array([int(y) for y in Y_test])

    head = torch.nn.Linear(base_model.transformer.wte.weight.shape[-1], max(Y_train)+1).to(device).to(dtype)

    param_groups = [
        {"params": [p for name, p in base_model.named_parameters() if "wte" in name], "lr": embed_lr},
        {"params": [p for name, p in base_model.named_parameters() if "wte" not in name], "lr": lr},
        {"params": head.parameters(), "lr": 1e-2}
    ]

    num_steps = int(num_epochs*len(X_train)/(batch_size*num_accumulation_steps))
    #optimizer = torch.optim.AdamW(list(base_model.parameters()) + list(head.parameters()), lr=lr)
    optimizer = torch.optim.AdamW(param_groups)
    #scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=num_steps)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=[embed_lr, lr, 1e-2], total_steps=num_steps, pct_start=0.05)
    loss_fn = torch.nn.CrossEntropyLoss()

    val_mcc = 0
    val_f1 = 0
    best_val_mcc = 0
    best_model_and_head = None
    total_loss = 0
    
    pbar = tqdm(range(num_steps))
    for step in pbar:
        if step % (num_steps // test_freq) == 0:
            base_model.eval()
            head.eval()
            with torch.no_grad():
                preds = []
                for i in range(0, len(X_val), batch_size):
                    pbar.set_description(f"(Testing {i}/{len(X_val)}) Loss: {total_loss:.4f}, val_mcc: {val_mcc*100:.2f}, val_f1: {val_f1*100:.2f}")
                    y = torch.tensor(Y_val[i:i+batch_size], dtype=torch.long, device=device)
                    x_subset = X_val[i:i+batch_size]

                    lens = []
                    x_tokenized = []
                    for sequence in x_subset:
                        tokenized = sp.encode("<DNA>" + sequence) + [EOS_TOKEN]
                        tokenized = [t for t in tokenized if t not in banned_tokens]
                        x_tokenized.append(tokenized)
                        lens.append(len(tokenized))
                    
                    if batch_size != 1:
                        max_len = max(lens)
                        x = torch.zeros((len(x_subset), max_len), dtype=torch.long, device=device)
                        x.fill_(PAD_TOKEN)
                        for i in range(0, len(x_tokenized)):
                            x[i][:lens[i]] = torch.tensor(x_tokenized[i], dtype=torch.long, device=device)

                        attn_mask = torch.zeros((x.shape[0], x.shape[-1], x.shape[-1])).to(device)
                        attn_mask = pad_attn(attn_mask, x)
                        attn_mask = attn_mask.unsqueeze(1).expand(-1, base_model.transformer.h[0].attn.n_head, -1, -1).to(dtype)
                        embeddings = base_model(x, attn_mask=attn_mask, return_embeddings=True)[:, 0]
                        y_pred = head(embeddings)
                        preds += y_pred.argmax(dim=-1).cpu().numpy().tolist()
                    else:
                        x = torch.tensor(x_tokenized, dtype=torch.long, device=device)
                        embeddings = base_model(x, return_embeddings=True)[:, 0]
                        y_pred = head(embeddings)
                        preds.append(y_pred.argmax(dim=-1).cpu().numpy().tolist())

                val_mcc = matthews_corrcoef(Y_val, preds)
                val_f1 = f1_score(Y_val, preds, average="weighted")

                if val_mcc > best_val_mcc:
                    best_val_mcc = val_mcc
                    best_model_and_head = (copy.deepcopy(base_model.state_dict()), copy.deepcopy(head.state_dict()))

            base_model.train()
            head.train()
        
        base_model.train()
        head.train()
        total_loss = 0
        for _ in range(num_accumulation_steps):
            indices = np.random.choice(len(X_train), batch_size, replace=False)
            y = torch.tensor(Y_train[indices], dtype=torch.long, device=device)

            ########## Prepare batch ##########
            lens = []
            x_tokenized = []
            for idx in indices:
                tokenized = sp.encode("<DNA>" + X_train[idx]) + [EOS_TOKEN]
                tokenized = [t for t in tokenized if t not in banned_tokens]
                x_tokenized.append(tokenized)
                lens.append(len(tokenized))

            if batch_size != 1:
                max_len = max(lens)
                x = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)
                x.fill_(PAD_TOKEN)
                for i in range(0, len(x_tokenized)):
                    x[i][:lens[i]] = torch.tensor(x_tokenized[i], dtype=torch.long, device=device)
                
                #attn_mask = create_attention_mask(x, base_model.transformer.h[0].attn.n_head).to(dtype).to(device)
                attn_mask = torch.zeros((x.shape[0], x.shape[-1], x.shape[-1])).to(device)
                attn_mask = pad_attn(attn_mask, x)
                attn_mask = attn_mask.unsqueeze(1).expand(-1, base_model.transformer.h[0].attn.n_head, -1, -1).to(dtype)
            ###################################

            optimizer.zero_grad()
            if batch_size != 1:
                embeddings = base_model(x, attn_mask=attn_mask, return_embeddings=True)[:, 0]
            else:
                x = torch.tensor(x_tokenized, dtype=torch.long, device=device)
                embeddings = base_model(x, return_embeddings=True)[:, 0]

            y_pred = head(embeddings)
            loss = loss_fn(y_pred, y) / num_accumulation_steps
            loss.backward()

            total_loss += loss.item()
        
        optimizer.step()
        scheduler.step()

        pbar.set_description(f"Loss: {total_loss:.4f}, val_mcc: {val_mcc*100:.2f}, val_f1: {val_f1*100:.2f}")
    
    pbar.close()

    base_model.eval()
    head.eval()
    with torch.no_grad():
        preds = []
        for i in range(0, len(X_test), batch_size):
            y = torch.tensor(Y_test[i:i+batch_size], dtype=torch.long, device=device)
            x_subset = X_test[i:i+batch_size]

            lens = []
            x_tokenized = []
            for sequence in x_subset:
                tokenized = sp.encode("<DNA>" + sequence) + [EOS_TOKEN]
                tokenized = [t for t in tokenized if t not in banned_tokens]
                x_tokenized.append(tokenized)
                lens.append(len(tokenized))
            
            if batch_size != 1:
                max_len = max(lens)
                x = torch.zeros((len(x_subset), max_len), dtype=torch.long, device=device)
                x.fill_(PAD_TOKEN)
                for i in range(0, len(x_tokenized)):
                    x[i][:lens[i]] = torch.tensor(x_tokenized[i], dtype=torch.long, device=device)

                #attn_mask = create_attention_mask(x, base_model.transformer.h[0].attn.n_head).to(dtype).to(device)
                attn_mask = torch.zeros((x.shape[0], x.shape[-1], x.shape[-1])).to(device)
                attn_mask = pad_attn(attn_mask, x)
                attn_mask = attn_mask.unsqueeze(1).expand(-1, base_model.transformer.h[0].attn.n_head, -1, -1).to(dtype)

                embeddings = base_model(x, attn_mask=attn_mask, return_embeddings=True)[:, 0]
            else:
                x = torch.tensor(x_tokenized, dtype=torch.long, device=device)
                embeddings = base_model(x, return_embeddings=True)[:, 0]

            y_pred = head(embeddings)
            preds += y_pred.argmax(dim=-1).cpu().numpy().tolist()
        
        test_mcc = matthews_corrcoef(Y_test, preds)
        test_f1 = f1_score(Y_test, preds, average="weighted")
        print(f"Test MCC: {test_mcc*100:.2f}, Test F1: {test_f1*100:.2f}")

    base_model.load_state_dict(best_model_and_head[0])
    head.load_state_dict(best_model_and_head[1])

    base_model.eval()
    head.eval()
    with torch.no_grad():
        preds = []
        for i in range(0, len(X_test), batch_size):
            y = torch.tensor(Y_test[i:i+batch_size], dtype=torch.long, device=device)
            x_subset = X_test[i:i+batch_size]

            lens = []
            x_tokenized = []
            for sequence in x_subset:
                tokenized = sp.encode("<DNA>" + sequence) + [EOS_TOKEN]
                tokenized = [t for t in tokenized if t not in banned_tokens]
                x_tokenized.append(tokenized)
                lens.append(len(tokenized))
            
            if batch_size != 1:
                max_len = max(lens)
                x = torch.zeros((len(x_subset), max_len), dtype=torch.long, device=device)
                x.fill_(PAD_TOKEN)
                for i in range(0, len(x_tokenized)):
                    x[i][:lens[i]] = torch.tensor(x_tokenized[i], dtype=torch.long, device=device)

                #attn_mask = create_attention_mask(x, base_model.transformer.h[0].attn.n_head).to(dtype).to(device)
                attn_mask = torch.zeros((x.shape[0], x.shape[-1], x.shape[-1])).to(device)
                attn_mask = pad_attn(attn_mask, x)
                attn_mask = attn_mask.unsqueeze(1).expand(-1, base_model.transformer.h[0].attn.n_head, -1, -1).to(dtype)
                embeddings = base_model(x, attn_mask=attn_mask, return_embeddings=True)[:, 0]
            else:
                x = torch.tensor(x_tokenized, dtype=torch.long, device=device)
                embeddings = base_model(x, return_embeddings=True)[:, 0]
            y_pred = head(embeddings)
            preds += y_pred.argmax(dim=-1).cpu().numpy().tolist()
        
        test_mcc = matthews_corrcoef(Y_test, preds)
        test_f1 = f1_score(Y_test, preds, average="weighted")
        print(f"Test MCC: {test_mcc*100:.2f}, Test F1: {test_f1*100:.2f}")

    return test_mcc, test_f1

def main(sp_dir, model_dir, banned_token, pretraining_epochs=4, pretraining_num_accum_steps=4, batch_size=32, pretraining_lr=1e-3, finetuning_lr=1e-3, output_suffix=""):
    print(f"Loading tokenizer from {sp_dir}...")
    print(f"Loading model from {model_dir}...")
    print(f"Using banned token {banned_token}")
    print(f"Pretraining for {pretraining_epochs} epochs with {pretraining_num_accum_steps} accumulation steps, batch size {batch_size}, lr {pretraining_lr}")
    print(f"Finetuning with lr {finetuning_lr}")
    print(f"Saving with output suffix: {output_suffix}")

    base_dir = "../datasets/GUE"
    all_subdirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    tasks = [os.path.join(all_subdirs[i], subdir) for i in range(len(all_subdirs)) for subdir in os.listdir(all_subdirs[i])]
    dtype = torch.bfloat16

    sp = spm.SentencePieceProcessor()
    sp.Load(sp_dir)
    banned_tokens = [banned_token]

    model = torch.load(model_dir, map_location=device).to(dtype).to(device)
    model.eval()

    print(f"Num params: {model.get_num_params() / 10**6:.2f}M")

    ########################### pre-train on sequence data ###########################
    all_sequences = []
    for task in tasks:
        X_train = load_task(task)[0]
        all_sequences += X_train
    
    all_sequences_tokenized = [sp.EncodeAsIds("<DNA>" + sequence) for sequence in tqdm(all_sequences)]
    for i in tqdm(range(0, len(all_sequences_tokenized))):
        all_sequences_tokenized[i].append(EOS_TOKEN)
        for banned_token in banned_tokens:
            while banned_token in all_sequences_tokenized[i]:
                all_sequences_tokenized[i].remove(banned_token)
    
    loss_hist = []
    num_epochs = pretraining_epochs
    num_accumulation_steps = pretraining_num_accum_steps
    lr = pretraining_lr
    embed_lr = 1e-3

    num_steps = int(num_epochs*len(all_sequences_tokenized)/(batch_size*num_accumulation_steps))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=num_steps)
    pbar = tqdm(range(num_steps))

    for step in pbar:
        optimizer.zero_grad()
        total_loss = 0
        for _ in range(num_accumulation_steps):
            indices = np.random.choice(len(all_sequences_tokenized), batch_size, replace=False)
            x = torch.ones((batch_size, max([len(all_sequences_tokenized[i]) for i in indices])), dtype=torch.long, device=device) * PAD_TOKEN
            for i, idx in enumerate(indices):
                x[i][:len(all_sequences_tokenized[idx])] = torch.tensor(all_sequences_tokenized[idx], dtype=torch.long, device=device)
            
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
        if "EMP" in task:
            epochs = 32
        elif "mouse" in task:
            epochs = 100
        elif "covid" in task:
            epochs = 32
        elif "tata" in task:
            epochs = 32
        elif "notata" in task:
            epochs = 32
        elif "all" in task:
            epochs = 32
        elif "splice" in task:
            epochs = 32
        elif "tf" in task:
            epochs = 32
        else:
            raise ValueError("Unknown task")

        print("---------------------------------------------------------------")
        print(f"Evaluting task {task}, training for {epochs} epochs...")
        mcc, f1 = finetune_on_task(task, model, sp, banned_tokens, device, dtype=dtype, batch_size=batch_size, num_accumulation_steps=num_accumulation_steps, num_epochs=epochs, lr=finetuning_lr, embed_lr=embed_lr, test_freq=100)
        results_ft[task] = {"mcc": mcc, "f1": f1}
        print("---------------------------------------------------------------")
    
    with open(f"GUE_results_{output_suffix}.csv", "w") as f:
        f.write("Task,MCC,F1\n")
        for task in results_ft:
            f.write(f"{task},{results_ft[task]['mcc']},{results_ft[task]['f1']}\n")

if __name__ == "__main__":
    fire.Fire(main)