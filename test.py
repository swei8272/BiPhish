import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from utils import load_sentence_polarity
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm.auto import tqdm
from model import CNN  # [FIX] Import from model.py

embedding_dim = 128
num_class = 2
batch_size = 256

def collate_fn(examples):
    inputs = [torch.tensor(ex[0]) for ex in examples]
    targets = torch.tensor([ex[1] for ex in examples], dtype=torch.long)
    inputs = pad_sequence(inputs, batch_first=True)
    return inputs, targets

class CnnDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, i):
        return self.data[i]

train_data, val_data, test_data, vocab = load_sentence_polarity(100)

train_dataset = CnnDataset(train_data)
val_dataset = CnnDataset(val_data)
test_dataset = CnnDataset(test_data)

train_data_loader = DataLoader(train_dataset, batch_size=256, collate_fn=collate_fn, shuffle=True)
val_data_loader = DataLoader(val_dataset, batch_size=1, collate_fn=collate_fn, shuffle=False)
test_data_loader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn, shuffle=False)

# [FIX] Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = torch.load("../model_train/model-cc_1.pkl", map_location=device)
model = model.to(device)
model.eval()

TP=FP=TN=FN=0
for batch in tqdm(test_data_loader, desc=f"Testing"):
    inputs, targets = [x.to(device) for x in batch]
    log_probs = model(inputs)

    pre = log_probs.argmax(dim=1).cpu().numpy()[0]
    label = targets.cpu().numpy()[0]
    if pre == 1 and label == 1: TP += 1
    if pre == 1 and label == 0: FP += 1
    if pre == 0 and label == 1: FN += 1
    if pre == 0 and label == 0: TN += 1

acc = (TP + TN) / (TP + FP + TN + FN)
pre = TP / (TP + FP) if (TP + FP) > 0 else 0
rec = TP / (TP + FN) if (TP + FN) > 0 else 0
f1s = (2 * pre * rec) / (pre + rec) if (pre + rec) > 0 else 0
print(f"TP = {TP}, TN = {TN}, FP = {FP}, FN = {FN}, accuracy = {acc}, precision = {pre}, recall = {rec}, F1 score = {f1s}")