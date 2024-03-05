"""Basic Tokenizer class implementation"""

from .tokenizer_base import BaseTokenzier


class Tokenizer(BaseTokenzier):
    """Basic tokenizer implementation"""

    @staticmethod
    def get_stats(ids):
        """"
        Count how frequently pairs occur in ids
        
        Parameters:
        - ids (List[int]): Encoded text data

        Returns:
        - stats (dict): frequency counts of pairs
        """

        stats = {}
        # count all pair occurances
        for id1, id2 in zip(ids, ids[1:]):
            stats[(id1, id2)] = stats.get((id1, id2), 0) + 1

        return stats

    @staticmethod
    def replace_pair(ids, pair, new_token):
        """
        Replace a pair of tokens with one new token. 
        
        Parameters:
        - ids (List[int]): Encoded text data.
        - pair (tuple(int,int)): Pair to be replaced.
        - new_token (int): New token to replace pair.

        Returns:
        - new_ids (List[int]): Encoded text data after replacement
        """

        if len(ids) <= 1:
            raise ValueError("Not enough tokens to merge")

        i = 0
        new_ids = []
        while i < len(ids):
            # If the pair matches then replace it with new token_id
            if i < len(ids) - 1 and (ids[i], ids[i+1]) == pair:
                new_ids.append(new_token)
                i += 2
            # If no match then just append current token to new list
            else:
                new_ids.append(ids[i])
                i += 1

        return new_ids

    def train(self, text, vocab_size):
        """
        Train the encoder from text data.

        Parameters:
        - text (str): 
        - vocab_size (int):
        """

        assert vocab_size >= 256
        iterations = vocab_size - 256

        ids = list(text.encode("utf-8"))

        for i in range(iterations):
            new_token = 256 + i
            print(f"iteration: {i}, {len(ids)}")
            stats = self.get_stats(ids)
            most_frequent_pair = min(stats, key = lambda k : stats.get(k, float("inf")))

            self.token_mapping[most_frequent_pair] = new_token
            ids = self.replace_pair(ids, most_frequent_pair, new_token)


        self.vocab = { idx : bytes([idx]) for idx in range(256) }

        for (p1, p2), idx in self.token_mapping.items():
            self.vocab[idx] = self.vocab[p1] + self.vocab[p2]


    def encode(self, text):
        """
        Encode a given string

        Parameters:
        - text (str): 
        """

        ids = list(text.encode("utf-8"))

        merging = True
        while merging:
            merging = False
            i = 0
            new_ids = []
            while i < len(ids):
                if i + 1 < len(ids) and (ids[i], ids[i+1]) in self.token_mapping:
                    merging = True
                    token = self.token_mapping[(ids[i], ids[i+1])]
                    new_ids.append(token)
                    i += 2
                else:
                    new_ids.append(ids[i])
                    i += 1
            ids = new_ids

        return new_ids

    def decode(self, ids):
        """
        Decoding strings
        
        Parameters:
        - ids (List[int])
        """

        byte_string = b"".join(self.vocab[idx] for idx in ids)
        decoded = byte_string.decode("utf-8", errors = 'replace')
        return decoded
