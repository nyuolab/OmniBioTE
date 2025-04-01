import torch
from model import OmniBioTA, OmniBioTAConfig
import argparse

if __name__ == "__main__":
    device = "cpu"

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_layer", type=int, default=12)
    parser.add_argument("--n_head", type=int, default=None)
    parser.add_argument("--n_embd", type=int, default=None)
    parser.add_argument("--ctx_len", type=int, default=1024)
    parser.add_argument("--vocab_size", type=int, default=4096)
    args = parser.parse_args()

    if args.n_head is None:
        args.n_head = args.n_layer
    if args.n_embd is None:
        args.n_embd = args.n_head * 128

    config = OmniBioTAConfig()
    config.vocab_size = args.vocab_size
    config.block_size = args.ctx_len
    config.n_embd = args.n_embd
    config.n_layer = args.n_layer
    config.n_head = args.n_head
    config.flash = True

    model = OmniBioTA(config)