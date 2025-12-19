import string

all_chars = string.ascii_letters + string.digits + " ,;.!?:/\|_@#$%^&*~`+-=<>()[]{}'\""

class Vocab:
    def __init__(self, tokens=None):
        self.idx_to_token = list()
        self.token_to_idx = dict()

        if tokens is not None:
            if "<unk>" not in tokens:
                tokens = tokens + ["<unk>"]
            for token in tokens:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token)
            self.unk = self.token_to_idx['<unk>']

    @classmethod
    def build(cls, text):
        uniq_tokens = ["<unk>"]
        uniq_tokens += list(all_chars)
        return cls(uniq_tokens)

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, token):
        return self.token_to_idx.get(token, self.unk)

    def convert_tokens_to_ids(self, tokens):
        return [self[token] for token in tokens]

    def convert_ids_to_tokens(self, indices):
        return [self.idx_to_token[index] for index in indices]

def save_vocab(vocab, path):
    with open(path, 'w', encoding='utf-8') as writer:
        writer.write("\n".join(vocab.idx_to_token))

def read_vocab(path):
    with open(path, 'r', encoding='utf-8') as f:
        tokens = f.read().split('\n')
    return Vocab(tokens)