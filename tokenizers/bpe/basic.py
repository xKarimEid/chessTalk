"""
basic Tokenizer class implementation
"""


class Tokenizer:
    """Basic tokenizer"""

    def __init__(self):
        self.token_mapping = {}
        self.vocab = {}

    @staticmethod
    def get_stats(ids):
        """"Count how frequently pairs occur in ids"""
        stats = {}
        for id1, id2 in zip(ids, ids[1:]):
            stats[(id1, id2)] = stats.get((id1, id2), 0) + 1
        return stats

    @staticmethod
    def replace_pair(ids, pair, new_token):
        """Replace a pair of tokens with one new token. Return a new list of ids"""

        if len(ids) <= 2:
            raise ValueError("Ran out of tokens!")

        i = 0
        new_ids = []
        while i < len(ids):
            if i < len(ids) - 1 and (ids[i], ids[i+1]) == pair:
                new_ids.append(new_token)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1

        return new_ids

    def train_encoder(self, text, iterations, token_start_id = 256):
        """Train the encoder"""

        ids = list(text.encode("utf-8"))

        for i in range(iterations):
            new_token = token_start_id + i
            stats = self.get_stats(ids)
            most_frequent_pair = sorted(stats, key = lambda k : stats[k], reverse=True)[0]

            self.token_mapping[most_frequent_pair] = new_token
            ids = self.replace_pair(ids, most_frequent_pair, new_token)


        self.vocab = { idx : bytes([idx]) for idx in range(token_start_id) }

        for (p1, p2), idx in self.token_mapping.items():
            self.vocab[idx] = self.vocab[p1] + self.vocab[p2]


    def encode(self, string):
        """Encode a given string"""

        ids = list(string.encode("utf-8"))

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
        """Decoding strings"""

        byte_string = b"".join(self.vocab[idx] for idx in ids)
        decoded = byte_string.decode("utf-8", errors = 'replace')
        return decoded
