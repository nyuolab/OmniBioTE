import os
import re
import sentencepiece as spm

class SingleCharTokenizer:
    def __init__(self, tokenizer_type):
        assert tokenizer_type in ["nuc", "prot"], "Invalid tokenizer type specified. Must be 'nuc' or 'prot'."
        if tokenizer_type == "prot":
            self.vocab = {
                'A': 23, 'R': 24, 'N': 25, 'D': 26, 'C': 27, 'Q': 28, 'E': 29, 
                'G': 30, 'H': 31, 'I': 32, 'L': 33, 'K': 34, 'M': 35, 'F': 36, 
                'P': 37, 'S': 38, 'T': 39, 'W': 40, 'Y': 41, 'V': 42, 'B': 43, 
                'Z': 44, 'X': 45, 'U': 46, 'O': 47,
                '<protein>': 48,
                '<EOS>': 3,
            }
        else:  # DNA
            self.vocab = {
                'A': 4, 'T': 5, 'C': 6, 'G': 7, 'N': 8,
                '<DNA>': 9, '<mRNA>': 10, '<RNA>': 11, '<rRNA>': 12, '<tRNA>': 13,
                '<cRNA>': 14, '<ss-RNA>': 15, '<ss-DNA>': 16, '<ds-mRNA>': 17,
                '<ds-rRNA>': 18, '<ds-RNA>': 19, '<ms-DNA>': 20, '<ms-RNA>': 21,
                '<ds-cRNA>': 22,
                '<EOS>': 3,
            }

        # Precompile a regex to capture <something> or a single char.
        # <[^>]*> means “less-than sign, anything not >, then >”
        self._pattern = re.compile(r'<[^>]*>|.')

    def Encode(self, sequence):
        vocab = self.vocab
        # Extract tokens in one pass:
        tokens_raw = self._pattern.findall(sequence)
        # Convert to IDs (default to UNKNOWN_TOKEN if not in vocab):
        return [vocab.get(tok, 0) for tok in tokens_raw]
        
    def Decode(self, token_ids):
        vocab = self.vocab
        # Convert IDs to tokens (default to UNKNOWN_TOKEN if not in vocab):
        tokens = [k for i in token_ids for k, v in vocab.items() if v == i]
        return "".join(tokens)

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

    def Encode(self, x):
        tokens = self.sp.EncodeAsIds(x)
        
        return [t + self.tokenizer_offset for t in tokens if t not in self.banned_tokens]

    def Decode(self, x):
        if not isinstance(x, list):
            x = [x]
        return self.sp.DecodeIds([t - self.tokenizer_offset for t in x])

class OmniTokenizer:
    def __init__(self, tokenizer_dir=None, single_char=False):
        if not single_char:
            assert tokenizer_dir is not None, "tokenizer_dir must be provided if using the SentencePiece tokenizers"
            nuc_sp = spm.SentencePieceProcessor(model_file=os.path.join(tokenizer_dir, 'genbank-2k.model'))
            prot_sp = spm.SentencePieceProcessor(model_file=os.path.join(tokenizer_dir, 'uniref-2k.model'))

            self.nuc_sp = TokenizerWrapper(nuc_sp, 0, [1, 2, 2037])
            self.prot_sp = TokenizerWrapper(prot_sp, 2048, [1, 2, 2044])
        else:
            self.nuc_sp = SingleCharTokenizer("nuc")
            self.prot_sp = SingleCharTokenizer("prot")

    def Encode(self, x, seq_type):
        assert seq_type in ['nuc', 'prot'], f"Invalid seq_type: {seq_type} (must be 'nuc' or 'prot')"
        
        if seq_type == 'nuc':
            return self.nuc_sp.Encode(x)
        else:
            return self.prot_sp.Encode(x)

    def Decode(self, x, seq_type):
        assert seq_type in ['nuc', 'prot'], f"Invalid seq_type: {seq_type} (must be 'nuc' or 'prot')"
        
        if seq_type == 'nuc':
            return self.nuc_sp.Decode(x)
        else:
            return self.prot_sp.Decode(x)