import argparse
import os
import sys
import json
import random
import numpy as np
import torch
import sentencepiece as spm
from tqdm import tqdm

# Insert path if needed for importing model definition
sys.path.insert(0, '../training')
from model import OmniBioTA

###############################################################################
# Global or shared settings
###############################################################################
device = "cuda:0" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16


###############################################################################
# Helper Classes and Functions
###############################################################################
class TokenizerWrapper:
    """
    A thin wrapper around the SentencePiece tokenizer that applies:
        - an offset to each token ID
        - skipping of banned tokens
    """
    def __init__(self, sp, tokenizer_offset, banned_tokens):
        self.sp = sp
        self.tokenizer_offset = tokenizer_offset
        self.banned_tokens = banned_tokens

    def EncodeAsIds(self, x):
        tokens = self.sp.EncodeAsIds(x)
        return [t + self.tokenizer_offset for t in tokens if t not in self.banned_tokens]

    def DecodeIds(self, x):
        if not isinstance(x, list):
            x = [x]
        return self.sp.DecodeIds([t - self.tokenizer_offset for t in x])
    
    def Decode(self, x):
        return self.DecodeIds(x)


def recurse_load(m):
    """
    Recursively find the original module if it is wrapped by a
    flash-attention-style `_orig_mod`.
    """
    if hasattr(m, "_orig_mod"):
        return recurse_load(m._orig_mod)
    return m


def load_model(fn):
    # Load base model
    base_model = torch.load(fn, map_location="cpu")
    base_model = recurse_load(base_model).to(device="cpu", dtype=dtype)

    model_sd = base_model.state_dict()
    config = base_model.config

    # Reinitialize OmniBioTA and load the saved state
    model = OmniBioTA(config).to(device="cpu", dtype=dtype)
    model.load_state_dict(model_sd)
    model.to(device=device, dtype=dtype)
    model.eval()

    del base_model
    return model


def get_motif(PFM):
    PFM = np.asarray(PFM)
    return "".join([["A", "C", "G", "T"][i] for i in PFM.argmax(axis=1)])


def generate_random_kmer(k=3):
    return "".join([random.choice("ACGT") for _ in range(k)])


def mutate_kmer(kmer, freq=0.1):
    return "".join([random.choice("ACGT") if random.random() < freq else nuc for nuc in kmer])


def get_complement(nuc_seq):
    return nuc_seq.translate(str.maketrans("ACGT", "TGCA"))[::-1]


@torch.no_grad()
def forward(model, head, stats, nuc_tokenizer, tokenized_prot_sequence, nuc_seq,
            append_complement=True, control=False):
    """
    Runs the forward pass through the model (and optionally includes complement).
    If `control` is True, returns a random normal deviate based on `stats`.
    """
    if append_complement:
        comp = "<DNA>" + get_complement(nuc_seq) + "<EOS>"
    else:
        comp = ""
    tokenized = tokenized_prot_sequence + nuc_tokenizer.EncodeAsIds("<DNA>" + nuc_seq + "<EOS>" + comp)
    tokenized = torch.tensor(tokenized, device=device, dtype=torch.long).unsqueeze(0)

    if tokenized.shape[-1] > 1024:
        raise ValueError("Input sequence too long")

    if control:
        # Control path: return random normal deviate using stats as mean, std
        return np.random.normal(stats[0], stats[1])

    emb = model(tokenized, return_embeddings=True)[:, 0].float()
    value = head(emb) * stats[1] + stats[0]  # scale + shift with stats
    return value.item()


###############################################################################
# Main
###############################################################################
def main():
    parser = argparse.ArgumentParser(description="Refactored evaluation script")
    parser.add_argument("--genbank_model_path", type=str, default="../tokenizers/bpe/genbank-2k.model",
                        help="Path to the genbank SentencePiece model.")
    parser.add_argument("--uniref_model_path", type=str, default="../tokenizers/bpe/uniref-2k.model",
                        help="Path to the uniref SentencePiece model.")
    parser.add_argument("--banned_sequences_file", type=str,
                        default="../datasets/JASPAR/banned_jaspar_sequences.txt",
                        help="File containing banned sequences, one per line.")
    parser.add_argument("--jaspar_data_file", type=str,
                        default="../datasets/JASPAR/JASPAR2024_CORE_processed.json",
                        help="JSON file containing JASPAR data.")
    parser.add_argument("--model_base_dir", type=str, default="",
                        help="Base directory for models and heads.")
    parser.add_argument("--folds", type=int, default=10, help="Number of folds to evaluate.")
    parser.add_argument("--replicates", type=int, default=8, help="Number of replicates per motif.")
    parser.add_argument("--output_suffix", type=str, default="", help="Suffix to append to output files.")
    parser.add_argument("--mutation_rates", type=str, default="0.05,0.1,0.25,0.5",
                        help="Comma-separated list of mutation rates to test.")
    args = parser.parse_args()

    # Prepare tokenizers
    nuc_sp = spm.SentencePieceProcessor(args.genbank_model_path)
    prot_sp = spm.SentencePieceProcessor(args.uniref_model_path)

    # Example offsets/banned tokens (adjust to match your use case)
    nuc_tokenizer = TokenizerWrapper(nuc_sp, 0, [1, 2, 2037])
    prot_tokenizer = TokenizerWrapper(prot_sp, 2048, [1, 2, 2044])

    # Load the banned sequences
    with open(args.banned_sequences_file, "r") as f:
        banned_seqs = list(f.read().splitlines())

    # Load JASPAR data, remove banned sequences
    with open(args.jaspar_data_file, "r") as f:
        data = json.load(f)
    data = [pair for pair in data if pair["sequence"] not in banned_seqs]

    # Convert comma-separated mutation rates to float
    mutation_rates = [float(x.strip()) for x in args.mutation_rates.split(",")]
    results = {rate: [] for rate in mutation_rates}

    # Evaluate for each fold
    for fold in range(args.folds):
        model_path = os.path.join(args.model_base_dir, f"model_fold{fold}.pt")
        head_path = os.path.join(args.model_base_dir, f"head_fold{fold}.pt")
        stats_path = os.path.join(args.model_base_dir, "all_stats.npy")

        model = load_model(model_path)
        head = torch.load(head_path, map_location="cpu").to(device=device, dtype=torch.float32)
        stats = np.load(stats_path) # mean, std

        for mutation_rate in results.keys():
            # Set random seeds for reproducibility
            np.random.seed(0)
            random.seed(0)

            # Optionally define how you get the "lowest probability motif"
            # This example just inlines the logic
            def get_lowest_prob_motif(PFM):
                PFM = np.asarray(PFM)
                return "".join([["A", "C", "G", "T"][i] for i in PFM.argmin(axis=1)])

            use_random_motif = False
            use_mutation = True
            control = False
            banned_sequences_encountered = 0

            pbar = tqdm(data, desc=f"Fold {fold}, mutation_rate={mutation_rate}")
            motif_dgs = []
            non_motif_dgs = []
            diffs = []

            for pair_of_interest in pbar:
                original_motif = get_motif(pair_of_interest["PFM"])
                sequence = pair_of_interest["sequence"]
                motif_len = len(pair_of_interest["PFM"])

                if sequence is None:
                    continue

                tokenized_prot_sequence = prot_tokenizer.EncodeAsIds("<protein>" + sequence + "<EOS>")
                if len(tokenized_prot_sequence) > 1000:
                    continue

                if sequence in banned_seqs:
                    banned_sequences_encountered += 1
                    continue

                # Evaluate the original motif
                delta_G = forward(model, head, stats, nuc_tokenizer,
                                  tokenized_prot_sequence, original_motif,
                                  append_complement=True, control=control)

                unique_motifs = set()
                motif = original_motif
                for _ in range(args.replicates):
                    # Find a mutated (or random) motif different from the original
                    while motif == original_motif or motif in unique_motifs:
                        if use_random_motif:
                            motif = generate_random_kmer(motif_len)
                        elif use_mutation:
                            motif = mutate_kmer(original_motif, mutation_rate)
                        else:
                            motif = get_lowest_prob_motif(pair_of_interest["PFM"])

                    unique_motifs.add(motif)

                    # Evaluate the mutated / non-motif
                    delta_G_non_motif = forward(model, head, stats, nuc_tokenizer,
                                                tokenized_prot_sequence, motif,
                                                append_complement=True, control=control)

                    motif_dgs.append(delta_G)
                    non_motif_dgs.append(delta_G_non_motif)
                    diffs.append(delta_G - delta_G_non_motif)

                    pbar.set_description(
                        f"Fold {fold}, rate={mutation_rate} | "
                        f"Mean motif ΔG: {np.mean(motif_dgs):.3f}, "
                        f"Mean non-motif ΔG: {np.mean(non_motif_dgs):.3f}, "
                        f"Mean diff: {np.mean(diffs):.3f}"
                    )

            results[mutation_rate].append(diffs)

    # Print final results
    for mutation_rate, all_diffs in results.items():
        means = [np.mean(d) for d in all_diffs if len(d) > 0]
        if len(means) == 0:
            print(f"Mutation rate: {mutation_rate} -> No valid data.")
            continue

        mean_of_means = np.mean(means)
        std_of_means = np.std(means) / np.sqrt(len(means))
        print(f"Mutation rate: {mutation_rate}: {mean_of_means:.4f} ± {std_of_means:.4f} (std err, n={len(means)})")
    
    with open(f"motif-selectivity-{args.output_suffix}.json", "w") as f:
        json.dump(results, f)

if __name__ == "__main__":
    main()