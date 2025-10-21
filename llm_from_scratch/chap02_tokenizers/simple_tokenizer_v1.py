
import re

# A simple tokenizer that splits text into tokens based on whitespace and punctuation.
class SimpleTokenizerV1:
    # Initializes the tokenizer with a given vocabulary
    def __init__(self, vocab):
        self.vocab = vocab
        self.inverse_vocab = {index: token for token, index in self.vocab.items()}

    # Encodes a string into a list of token IDs
    def encode(self, text):
        tokens = re.split(r'([,.:;?_!"()\'`]|--|\s)', text)
        tokens = [tok.strip() for tok in tokens if tok.strip()]
        print("encoded tokens:", tokens)
        return [self.vocab[token] for token in tokens]

    # Decodes a list of token IDs back into a string
    def decode(self, token_ids):
        text = ' '.join([self.inverse_vocab[token_id] for token_id in token_ids])
        text = re.sub(r'\s+([,.:;?_!"()\'`])', r'\1', text)
        return text
