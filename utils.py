import torch
from vocab import Vocab, save_vocab
import numpy as np
from nltk.corpus import treebank

# parameter
Phishing_url_data_path = r"./data/phish-58w.txt"
Legitimate_url_data_path = r"./data/legtimate-58w.txt"
train_prop = 0.7
val_prop = 0.1
test_prop = 0.2


def get_char_ngrams(word):
    chars = list(word)
    begin_idx = 0
    ngrams = []
    while (begin_idx + 1) <= len(chars):
        end_idx = begin_idx + 1
        ngrams.append("".join(chars[begin_idx:end_idx]))
        begin_idx += 1
    return ngrams


def load_data(filepath):
    data = []
    data_char = []
    with open(filepath, encoding="utf-8") as f:
        for line in f.readlines():
            # [FIXED] Removed print(line) for speed
            data.append(line.strip('\n'))

    for line in data:
        data_char.append(get_char_ngrams(line))

    return data_char


def url_cut(url, length):
    cut = []
    for line in url:
        cut.append(line[:length])
    return cut


def load_sentence_polarity(length):
    Phishing_url_data = load_data(Phishing_url_data_path)
    Phishing_url_data = url_cut(Phishing_url_data, length)

    p_len = len(Phishing_url_data)
    Phishing_url_data_test = Phishing_url_data[:int(test_prop * p_len)]
    Phishing_url_data_val = Phishing_url_data[int(test_prop * p_len):int((test_prop + val_prop) * p_len)]
    Phishing_url_data_train = Phishing_url_data[int((test_prop + val_prop) * p_len):]

    Legitimate_url_data = load_data(Legitimate_url_data_path)
    Legitimate_url_data = url_cut(Legitimate_url_data, length)

    l_len = len(Legitimate_url_data)
    Legitimate_url_data_test = Legitimate_url_data[:int(test_prop * l_len)]
    Legitimate_url_data_val = Legitimate_url_data[int(test_prop * l_len):int((test_prop + val_prop) * l_len)]
    Legitimate_url_data_train = Legitimate_url_data[int((test_prop + val_prop) * l_len):]

    vocab = Vocab.build(Phishing_url_data + Legitimate_url_data)
    save_vocab(vocab, './vocab.txt')

    train_data = [(vocab.convert_tokens_to_ids(sentence), 0) for sentence in Phishing_url_data_train] \
                 + [(vocab.convert_tokens_to_ids(sentence), 1) for sentence in Legitimate_url_data_train]

    val_data = [(vocab.convert_tokens_to_ids(sentence), 0) for sentence in Phishing_url_data_val] \
               + [(vocab.convert_tokens_to_ids(sentence), 1) for sentence in Legitimate_url_data_val]

    test_data = [(vocab.convert_tokens_to_ids(sentence), 0) for sentence in Phishing_url_data_test] \
                + [(vocab.convert_tokens_to_ids(sentence), 1) for sentence in Legitimate_url_data_test]

    return train_data, val_data, test_data, vocab


def length_to_mask(lengths):
    max_len = torch.max(lengths)
    mask = torch.arange(max_len).expand(lengths.shape[0], max_len) < lengths.unsqueeze(1)
    return mask


def load_treebank():
    sents, postags = zip(*(zip(*sent) for sent in treebank.tagged_sents()))
    vocab = Vocab.build(sents, reserved_tokens=["<pad>"])
    tag_vocab = Vocab.build(postags)

    train_data = [(vocab.convert_tokens_to_ids(sentence), tag_vocab.convert_tokens_to_ids(tags))
                  for sentence, tags in zip(sents[:3000], postags[:3000])]
    test_data = [(vocab.convert_tokens_to_ids(sentence), tag_vocab.convert_tokens_to_ids(tags))
                 for sentence, tags in zip(sents[3000:], postags[3000:])]

    return train_data, test_data, vocab, tag_vocab


class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss