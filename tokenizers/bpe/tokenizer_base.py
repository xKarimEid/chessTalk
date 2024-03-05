"""Implements signature of a basic tokenizer class"""

class BaseTokenzier:
    """
    Basic tokenizer functionality
    """

    def __init__(self):
        self.token_mapping = {}
        self.vocab = self._build_vocab()

    def train(self, text, vocab_size):
        """
        Train the tokenizer on the given text. 

        Parameters: 
        - text (str): Text data for training
        - vocab_size (int): Size of the vocabularly.
        """
        raise NotImplementedError

    def encode(self, text):
        """
        Encode the given text. 

        Parameters:
        - text (str): Text data to encode.
        """
        raise NotImplementedError

    def decode(self, ids):
        """
        Decode the given string.

        Parameters:
        - ids (list): List of IDs to decode.
        """
        raise NotImplementedError

    def _build_vocab(self):
        """
        Build the vocabularly based on token mappings.
        """

        vocab = {idx : bytes([idx]) for idx in range(256)}

        for (p1, p2), idx in self.token_mapping.items():
            vocab[idx] = vocab[p1] + vocab[p2]

        return vocab

    def save(self, path = 'tokenizers/models/'):
        """
        Save the tokenizer model to file.

        Parameters:
        - path (str): Path to save the model file.
        """

        model_file = path + '.model'
        with open(model_file, 'w', encoding='utf-8') as file:
            file.write("Basic encoder v1\n")

            for idx1, idx2 in self.token_mapping:
                file.write(f"{idx1} {idx2}\n")

    def load(self, path = 'tokenizers/models/'):
        """
        Load the tokenizer model.

        Parameters:
        - path (str): Path to the model file.
        """

        # Reset token_mapping
        self.token_mapping = {}
        # start of new token values
        new_token = 256

        model_file = path + '.model'
        with open(model_file, "r", encoding="utf-8") as file:
            # First line contains model version
            version = file.readline()
            print(version)

            # Remaining lines contain the token pairs
            for line in file.readlines():
                ix1, ix2 = map(int, line.split(" "))
                self.token_mapping[(ix1, ix2)] = new_token
                new_token += 1

        # Build vocab using the new tokens
        self.vocab = self._build_vocab()
