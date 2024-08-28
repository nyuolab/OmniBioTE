import os
import numpy as np
from tqdm import tqdm
import torch
from loader import line_reader, get_sequence, get_batch, EOS_TOKEN, PAD_TOKEN, data_loader_parallel
from model import OmniBioTA, OmniBioTAConfig
from mup import set_base_shapes, MuAdamW
import argparse
import threading
import queue
import json
import time
import socket

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

MASK_TOKEN = 2
dtype = torch.bfloat16

torch.backends.cudnn.benchmark = True # enable cuDNN benchmarking for faster training

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

def run(args):
    # data config

    ###### TOKENIZER FIX #######
    # for peptide_bpe, the banned token is 65530
    # for nucleotide_bpe, the banned token is 65525
    # for mixed_bpe, the banned token is 65533
    banned_tokens = [args.banned_token] # this represents the "_" token, which for some reason showed up in the tokenizer
    ############################

    ########### INITIALIZE DATALOADERS #################
    base_dir = args.base_dir

    if args.train_type == "protein":
        train_dirs = ["uniref100/train"]
        test_dirs = ["uniref100/val"]
        test_names = ["uniref100"]
        train_proportion = [1.0]
    elif args.train_type == "nucleotide":
        train_dirs = ["genbank/train"]
        test_dirs = ["genbank/val"]
        test_names = ["genbank"]
        train_proportion = [1.0]
    elif args.train_type == "mixed":
        train_dirs = ['genbank/train', 'uniref100/train'] # the directories to pull training data from
        test_dirs = ['genbank/val', 'uniref100/val'] # the directories to pull test data from
        test_names = ["genbank", "uniref100"] # the names of the test datasets
        train_proportion = [0.80, 0.20] # the proportion of each dataset size compared to the total dataset
    elif args.train_type == "halfnhalf":
        train_dirs = ['genbank/train', 'uniref100/train'] # the directories to pull training data from
        test_dirs = ['genbank/val', 'uniref100/val'] # the directories to pull test data from
        test_names = ["genbank", "uniref100"] # the names of the test datasets
        train_proportion = [0.50, 0.50] # the proportion of each dataset size compared to the total dataset
    else:
        raise ValueError("Invalid train_type. Must be one of 'protein', 'nucleotide', 'mixed', or 'halfnhalf'")
    
    train_dirs = [os.path.join(base_dir, train_dir) for train_dir in train_dirs] # add the base directory to the training directories
    train_files = [[os.path.join(train_dir, file) for file in os.listdir(train_dir)] for train_dir in train_dirs] # get all files in each directory

    test_dirs = [os.path.join(base_dir, test_dir) for test_dir in test_dirs] # add the base directory to the test directories
    test_files = [[os.path.join(test_dir, file) for file in os.listdir(test_dir)] for test_dir in test_dirs] # get all files in each directory

    ###################################################

    ########### INITIALIZE DISTRIBUTED TRAINING #################
    # initialize the process group
    dist.init_process_group("nccl")
    gloo_group = dist.new_group(backend="gloo")
    nccl_group = dist.new_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    device = f"cuda:{rank % torch.cuda.device_count()}"
    node_name = socket.gethostname()
    print(f"Running on rank {rank}, node {os.environ['SLURMD_NODENAME']}, device {device}, node name {node_name}")

    assert args.batch_size % world_size == 0, "Batch size must be divisible by the number of processes."
    
    logging = True # whether to log the training loss on wandb
    batch_size = args.batch_size // world_size # batch size per process
    ctx_len = args.ctx_len # context length of the model
    batch_split = [int(x * batch_size) for x in train_proportion] # number of samples to pull from each dataset for each batch

    # make sure the sum of train_ints is equal to batch_size
    if sum(batch_split) != batch_size:
        batch_split[-1] += batch_size - sum(batch_split)

    # train generator
    line_readers = [line_reader(train_file, banned_tokens=banned_tokens) for train_file in train_files] # get a line reader for each file group
    generators = [get_sequence(reader, ctx_len, args.use_padding) for reader in line_readers] # get a sequence generator for each line reader
    batch_generator = get_batch(generators, batch_split, return_pt=True) # get a batch generator for each sequence generator

    # test generator
    test_line_readers = [line_reader(test_file, banned_tokens=banned_tokens) for test_file in test_files]
    test_generators = [get_sequence(reader, ctx_len, args.use_padding) for reader in test_line_readers]
    ###########################################################

    last_test = 0 # the number of tokens that were elapsed on the previous test
    last_save = 0 # the number of tokens that were elapsed on the previous save

    # start a thread to load data in parallel
    batch_queue = queue.Queue(maxsize=2)
    loader_thread = threading.Thread(target=data_loader_parallel, args=(batch_queue, batch_generator, device))
    loader_thread.start()

    # set up the model
    config = OmniBioTAConfig()
    config.vocab_size = 2**16
    config.dropout = args.dropout
    config.block_size = args.ctx_len
    config.n_embd = args.n_embd
    config.n_layer = args.n_layer
    config.n_head = args.n_head
    config.flash = not args.disable_flash
    config.checkpoint_freq = args.checkpoint_freq

    m = OmniBioTA(config)

    # set up muP (see https://github.com/microsoft/mup)
    config.n_embd = 24
    config.n_head = 3
    base_model = OmniBioTA(config)

    config.n_embd = 48
    config.n_head = 12
    delta_model = OmniBioTA(config)

    set_base_shapes(m, base_model, delta=delta_model) # initialize model with muP

    del base_model, delta_model # delete the base and delta models to save memory

    m.to(dtype).to(device) # move the model to the device and set the dtype

    num_model_params = m.get_num_params()

    if args.resume_from > 0:
        m = torch.load(f"{args.save_name}_{args.resume_from}.pt", map_location=device)
        m.to(dtype).to(device)
        torch.cuda.empty_cache()
        print(f"Loaded model from {args.resume_from} token checkpoint")

    #m = torch.jit.script(m) # this isn't quite working yet
    if args.FSDP:
        torch.cuda.set_device(rank)
        model = FSDP(m)
    else:
        model = DDP(m, device_ids=[rank % torch.cuda.device_count()])

    #model = torch.compile(model) # compile the model for compute efficiency

    if logging and rank == 0:
        import wandb
        wandb.init(project=args.wandb_project_name)

    token_budget = args.token_budget # the number of tokens to train on
    total_iters = int(token_budget / (world_size * batch_size * ctx_len)) # the total number of iterations to train for
    lr = args.lr*np.sqrt(args.batch_size) / 32 # the learning rate to use, scaled by batch size (default batch size is 1024)
    if args.force_lr:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay, betas=(args.beta1, args.beta2), eps=args.epsilon) # initialize optimizer without muP
    else:
        optimizer = MuAdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay, betas=(args.beta1, args.beta2), eps=args.epsilon) # initialize optimizer with muP
    #scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=total_iters, pct_start=args.warmup_period)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=total_iters)
    mini_batch_size = args.mini_batch_size # the mini batch size for gradient accumulation
    
    effective_batch_size = mini_batch_size # this is for batch size ramp up
    grand_batch = batch_queue.get(block=True).to(device) # a large batch from which to pull batches (needed for this batch size ramp up implementation)

    trained_tokens = 0
    starting_step = 0

    if args.resume_from > 0:
        last_test = args.resume_from
        last_save = args.resume_from
        trained_tokens = args.resume_from

        remaining_steps = total_iters - int(total_iters * (trained_tokens / token_budget))

        optimizer = torch.load(f"{args.save_name}_optimizer_{args.resume_from}.pt", map_location=device)
        print(f"Loaded optimizer from {args.resume_from} token checkpoint")
        #scheduler = torch.load(f"{args.save_name}_scheduler_{args.resume_from}.pt", map_location=device)
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0 * trained_tokens / token_budget, end_factor=0.0, total_iters=remaining_steps)
        
    if args.resume_from > 0:
        starting_step = total_iters - remaining_steps
        '''
        # get LR from the scheduler and compute what step we're on
        lrs = []
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
            lrs.append(lr)
        
        ratio = 1 - lrs[1] / args.lr
        starting_step = int(total_iters * ratio)
        trained_tokens = args.resume_from
        '''
    
    if rank == 0:
        pbar = tqdm(range(starting_step, total_iters))
    else:
        pbar = range(starting_step, total_iters)
        
    for i in pbar:
        start_time = time.time()

        # compute the effective batch size for this iteration
        if args.batch_ramp:
            # compute the effective batch size for this iteration
            effective_batch_size = min((int(i / (total_iters * args.warmup_period) * batch_size) // mini_batch_size) * mini_batch_size + mini_batch_size, batch_size)
            
            # round to nearest multiple of mini_batch_size
            effective_batch_size = effective_batch_size // mini_batch_size * mini_batch_size
        else:
            effective_batch_size = batch_size
                
        # round to nearest multiple of mini_batch_size
        effective_batch_size = effective_batch_size // mini_batch_size * mini_batch_size

        
        if grand_batch.shape[0] < effective_batch_size: # if the grand batch is too small, pull more batches from the queue
            batch_start = time.time()
            grand_batch = torch.cat((grand_batch, batch_queue.get(block=True).to(device)), dim=0)
            batch_queue.task_done()
            batch_end = time.time()

            if rank == 0:
                wandb.log({"timing/batch_fetch_time": batch_end - batch_start}, step=trained_tokens)
        
        input_ids = grand_batch[:effective_batch_size] # get a batch size of effective_batch_size from the grand batch
        grand_batch = grand_batch[effective_batch_size:] # remove the batch from the grand batch

        cum_loss = 0 # cumulative loss for this iteration
        optimizer.zero_grad(set_to_none=True)

        mask_prob = 0.15 # probability of masking a token

        # mask tokens
        mask = np.random.binomial(1, mask_prob, input_ids.shape)
        mask = torch.as_tensor(mask, dtype=torch.bool, device=device)
        mask = mask & (input_ids != PAD_TOKEN) & (input_ids != EOS_TOKEN)
        masked_ids = input_ids.masked_fill(mask, MASK_TOKEN)
        
        mask_times = []
        forward_times = []
        backward_times = []
        for j in range(input_ids.shape[0] // mini_batch_size): # gradient accumulation
            mini_batch_x = masked_ids[j*mini_batch_size:(j+1)*mini_batch_size]
            mini_batch_y = input_ids[j*mini_batch_size:(j+1)*mini_batch_size]

            mask_start = time.time()
            # create attention mask
            attn_mask = torch.ones((args.mini_batch_size, args.ctx_len, args.ctx_len), device=device, dtype=dtype) * -1e9
            attn_mask = create_attention_mask(attn_mask, mini_batch_y, padding=args.use_padding)
            attn_mask = attn_mask.unsqueeze(1).expand(-1, args.n_head, -1, -1)
            mask_times.append(time.time() - mask_start)

            forward_start = time.time()
            logits = model.forward(mini_batch_x.view(mini_batch_size, -1), attn_mask=attn_mask)
            #embeddings = model.forward(mini_batch_x.view(mini_batch_size, -1), attn_mask=attn_mask, return_embeddings=True)
            #logits = model.module.lm_head(embeddings) # we do this to avoid torch.compile() errors with MuReadout()
            forward_times.append(time.time() - forward_start)

            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), mini_batch_y.view(-1), reduction="none") / (effective_batch_size // mini_batch_size)
            
            # ensure cross entropy is only computed for the masked tokens
            loss *= mask[j*mini_batch_size:(j+1)*mini_batch_size].view(-1).float()
            loss = loss.sum() / mask[j*mini_batch_size:(j+1)*mini_batch_size].view(-1).sum()

            backward_start = time.time()
            loss.backward()
            backward_times.append(time.time() - backward_start)

            cum_loss += loss.item()
        
        optimizer_start = time.time()

        # clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        optimizer_end = time.time()

        if rank == 0:
            wandb.log({"timing/mask_time": np.mean(mask_times), "timing/forward_time": np.mean(forward_times), "timing/backward_time": np.mean(backward_times), "timing/optimizer_time": optimizer_end - optimizer_start}, step=trained_tokens)

        # get current learning rate
        lrs = []
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
            lrs.append(lr)
        
        if len(lrs) == 1:
            lrs.append(lrs[0])
        
        all_cum_loss = [None for _ in range(world_size)]
        dist.all_gather_object(all_cum_loss, cum_loss, group=gloo_group)
        cum_loss = np.mean(all_cum_loss) # this technically isn't exactly the same as the loss (due to different padding lengths), but it's close enough
        
        if rank == 0:
            if trained_tokens < 1e6:
                pbar.set_description(f"Loss: {cum_loss:.4f}, LR: {lrs[0]:.4f} | {lrs[1]}, Trained Tokens: {trained_tokens/1e3:.3f}K")
            elif trained_tokens < 1e9:
                pbar.set_description(f"Loss: {cum_loss:.4f}, LR: {lrs[0]:.4f} | {lrs[1]}, Trained Tokens: {trained_tokens/1e6:.3f}M")
            else:
                pbar.set_description(f"Loss: {cum_loss:.4f}, LR: {lrs[0]:.4f} | {lrs[1]}, Trained Tokens: {trained_tokens/1e9:.3f}B")
            
            if logging:
                wandb.log({"loss": cum_loss, "lr": lrs[0], "batch_size": effective_batch_size * world_size}, step=trained_tokens)
        
        # count total number of tokens in the batch
        num_tokens = (input_ids != PAD_TOKEN).sum().item()

        all_num_tokens = [None for _ in range(world_size)]
        # the first argument is the collected lists, the second argument is the data unique in each process
        dist.all_gather_object(all_num_tokens, num_tokens, group=gloo_group)

        end_time = time.time()

        # compute efficiency relative to A100
        tokens_per_sec = sum(all_num_tokens) / (end_time - start_time)
        actual_flops = 6  * num_model_params + 12 * args.n_layer * args.n_embd * args.ctx_len # 6N + 12*LHQT estimate of flops per token
        actual_flops *= sum(all_num_tokens)
        flops_per_sec = actual_flops / (end_time - start_time)
        a100_flops_per_sec = 312e12 # A100 flops per second
        efficiency = flops_per_sec / (a100_flops_per_sec * world_size) * 100
        
        if rank == 0:
            wandb.log({"timing/tokens_per_sec": tokens_per_sec, "timing/total_train_step_time": end_time - start_time, "A100 efficiency": efficiency}, step=trained_tokens)

        trained_tokens += sum(all_num_tokens)

        if trained_tokens - last_test > args.test_freq:
            # test the model
            model.eval()

            with torch.no_grad():
                for test_generator, test_name in zip(test_generators, test_names):
                    test_batch = next(test_generator)
                    for _ in range(1, mini_batch_size):
                        test_batch = np.concatenate((test_batch, next(test_generator)))
                    test_batch = torch.as_tensor(test_batch, dtype=torch.long, device=device).view(mini_batch_size, -1)

                    mask = np.random.binomial(1, mask_prob, test_batch.shape)
                    mask = torch.as_tensor(mask, dtype=torch.bool, device=device)
                    mask = mask & (test_batch != PAD_TOKEN) & (test_batch != EOS_TOKEN)
                    masked_ids = test_batch.masked_fill(mask, MASK_TOKEN)

                    attn_mask = torch.ones((args.mini_batch_size, args.ctx_len, args.ctx_len), device=device, dtype=dtype) * -1e9
                    attn_mask = create_attention_mask(attn_mask, test_batch, padding=args.use_padding)
                    attn_mask = attn_mask.unsqueeze(1).expand(-1, args.n_head, -1, -1)

                    logits = model.forward(masked_ids, attn_mask=attn_mask)
                    #embeddings = model.forward(masked_ids, attn_mask=attn_mask, return_embeddings=True)
                    #logits = model.module.lm_head(embeddings)
                    loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), test_batch.view(-1), reduction="none")
                    loss *= mask.view(-1).float()
                    loss = loss.sum() / mask.view(-1).sum() / world_size

                    all_losses = [None for _ in range(world_size)]
                    dist.all_gather_object(all_losses, loss.item(), group=gloo_group)

                    all_num_tokens = [None for _ in range(world_size)]
                    dist.all_gather_object(all_num_tokens, mask.view(-1).sum().item(), group=gloo_group)

                    # log the test loss
                    if logging and rank == 0:
                        wandb.log({f"test_loss/{test_name}": float(np.sum(all_losses))}, step=trained_tokens)

            model.train()

            last_test = trained_tokens
        
        if rank == 0 and trained_tokens - last_save > args.save_freq:
            torch.save(model.module, f"{args.save_name}_{trained_tokens}.pt")
            torch.save(optimizer, f"{args.save_name}_optimizer_{trained_tokens}.pt")
            torch.save(scheduler, f"{args.save_name}_scheduler_{trained_tokens}.pt")
            
            if last_save > 0:
                os.remove(f"{args.save_name}_{last_save}.pt")
                os.remove(f"{args.save_name}_optimizer_{last_save}.pt")
                os.remove(f"{args.save_name}_scheduler_{last_save}.pt")


            last_save = trained_tokens
        
        total_loop_time = time.time() - start_time
        if rank == 0:
            wandb.log({"timing/total_loop_time": total_loop_time}, step=trained_tokens)
    
    if rank == 0:
        torch.save(model.module, f"{args.save_name}.pt")
        torch.save(optimizer, f"{args.save_name}_optimizer.pt")
        torch.save(scheduler, f"{args.save_name}_scheduler.pt")
        

    dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1024, help="The total batch size across all processes")
    parser.add_argument("--mini_batch_size", type=int, default=8, help="The batch size for gradient accumulation, and the batch size for each process")
    parser.add_argument("--n_head", type=int, default=8, help="The number of attention heads")
    parser.add_argument("--n_embd", type=int, default=1024, help="The embedding dimension")
    parser.add_argument("--n_layer", type=int, default=8, help="The number of transformer layers")
    parser.add_argument("--ctx_len", type=int, default=2048, help="The context length")
    parser.add_argument("--dropout", type=float, default=0.1, help="The dropout probability")
    parser.add_argument("--lr", type=float, default=1e-2, help="The learning rate (scaled by muP, unless --force_lr is set)")
    parser.add_argument("--beta1", type=float, default=0.9, help="The beta1 parameter for AdamW")
    parser.add_argument("--beta2", type=float, default=0.999, help="The beta2 parameter for AdamW")
    parser.add_argument("--epsilon", type=float, default=1e-8, help="The epsilon parameter for AdamW")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="The weight decay parameter for AdamW")
    parser.add_argument("--token_budget", type=float, default=20e9, help="The number of tokens to train on")
    parser.add_argument("--test_freq", type=int, default=1e7, help="The number of tokens between tests")
    parser.add_argument("--save_freq", type=int, default=1e9, help="The number of tokens between saves")
    parser.add_argument("--save_name", type=str, default="omnibiota", help="The prefix name to save the model as")
    parser.add_argument("--disable_flash", action="store_true", default=False, help="Whether to disable flash attention")
    parser.add_argument("--wandb_project_name", type=str, default="omnibiota", help="The name of the wandb project to log to")
    parser.add_argument("--base_dir", type=str, default="", help="The base directory for the training and validation data")
    parser.add_argument("--force_lr", action="store_true", default=False, help="Whether to override muP's learning rate scaling")
    parser.add_argument("--checkpoint_freq", type=int, default=0, help="The frequency at which activations are checkpointed")
    parser.add_argument("--banned_token", type=int, help="The token to ban from the tokenizer")
    parser.add_argument("--warmup_period", type=float, default=0.05, help="The proportion of the total iterations to warm up")
    parser.add_argument("--batch_ramp", action="store_true", default=False, help="Whether to ramp up the batch size")
    parser.add_argument("--train_type", type=str, default="mixed", help="The type of training to perform")
    parser.add_argument("--FSDP", action="store_true", default=False, help="Whether to use FullyShardedDataParallel")
    parser.add_argument("--use_padding", action="store_true", default=False, help="Whether to pad the sequence with PAD_TOKEN instead of truncating a line if it doesn't fit in")
    parser.add_argument("--resume_from", type=int, default=0, help="The token number to resume training from")
    args = parser.parse_args()

    run(args)