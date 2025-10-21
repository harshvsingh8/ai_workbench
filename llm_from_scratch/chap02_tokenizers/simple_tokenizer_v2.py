
import re

class SimpleTokenizerV2:
    # Initializes the tokenizer with a given vocabulary
    def __init__(self, vocab):
        vocab = vocab.copy()
        # Extend the vocabulary with a special unknown tokens (if not already present)
        if "<|unk|>" not in vocab:
            vocab["<|unk|>"] = len(vocab)
        if "<|endoftext|>" not in vocab:
            vocab["<|endoftext|>"] = len(vocab)
        self.vocab = vocab
        self.inverse_vocab = {index: token for token, index in self.vocab.items()}

    # Encodes a string into a list of token IDs
    def encode(self, text):
        tokens = re.split(r'([,.:;?_!"()\'`]|--|\s)', text)
        tokens = [tok.strip() for tok in tokens if tok.strip()]
        tokens = [tok if tok in self.vocab else "<|unk|>" for tok in tokens]
        if __debug__:
            print("encoded tokens:", tokens)
        return [self.vocab[token] for token in tokens]

    # Decodes a list of token IDs back into a string
    def decode(self, token_ids):
        text = ' '.join([self.inverse_vocab[token_id] for token_id in token_ids])
        text = re.sub(r'\s+([,.:;?_!"()\'`])', r'\1', text)
        return text
