import os
import warnings
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from utils import load_sentence_polarity, EarlyStopping
import numpy as np
from tqdm.auto import tqdm
from model import CNN

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['KMP_WARNINGS'] = '0'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

print("=" * 80)
print("CNN TRAINING FOR PHISHING DETECTION")
print("=" * 80)

class CnnDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, i):
        return self.data[i]

def collate_fn(examples):
    inputs = [torch.tensor(ex[0]) for ex in examples]
    targets = torch.tensor([ex[1] for ex in examples], dtype=torch.long)
    inputs = pad_sequence(inputs, batch_first=True)
    return inputs, targets

num_epoch = 50
embedding_dim = 128
num_class = 2

print("\n[1/6] Loading data...")
train_data, val_data, test_data, vocab = load_sentence_polarity(100)
print(f"✓ Training: {len(train_data)}, Validation: {len(val_data)}, Test: {len(test_data)}")

print("\n[2/6] Creating data loaders...")
train_dataset = CnnDataset(train_data)
val_dataset = CnnDataset(val_data)
test_dataset = CnnDataset(test_data)

train_data_loader = DataLoader(train_dataset, batch_size=256, collate_fn=collate_fn, shuffle=True)
val_data_loader = DataLoader(val_dataset, batch_size=1, collate_fn=collate_fn, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✓ Device: {device}")

print("\n[3/6] Building model...")
model = CNN(len(vocab) + 1, embedding_dim, num_class)
model.to(device)
nll_loss = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.000001)

train_losses = []
val_losses = []
early_stopping = EarlyStopping(patience=50, verbose=True)

print("\n[4/6] Training (this will take 30-60 minutes)...")
for epoch in range(1, num_epoch + 1):
    model.train()
    for batch in tqdm(train_data_loader, desc=f"Epoch {epoch}"):
        optimizer.zero_grad()
        inputs, targets = [x.to(device) for x in batch]
        log_probs = model(inputs)
        loss = nll_loss(log_probs, targets)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    model.eval()
    with torch.no_grad():
        for batch in val_data_loader:
            inputs, targets = [x.to(device) for x in batch]
            log_probs = model(inputs)
            loss = nll_loss(log_probs, targets)
            val_losses.append(loss.item())

    train_loss = np.average(train_losses)
    val_loss = np.average(val_losses)
    print(f'[{epoch}/{num_epoch}] train_loss: {train_loss:.5f} valid_loss: {val_loss:.5f}')

    train_losses = []
    val_losses = []

    early_stopping(val_loss, model)
    if early_stopping.early_stop:
        print("Early stopping!")
        break

print("\n[5/6] Evaluating...")
model.load_state_dict(torch.load('checkpoint.pt'))
model.eval()

TP = FP = TN = FN = 0
with torch.no_grad():
    for batch in val_data_loader:
        inputs, targets = [x.to(device) for x in batch]
        log_probs = model(inputs)
        pre = log_probs.argmax(dim=1).cpu().numpy()[0]
        label = targets.cpu().numpy()[0]
        if pre == 1 and label == 1: TP += 1
        if pre == 1 and label == 0: FP += 1
        if pre == 0 and label == 1: FN += 1
        if pre == 0 and label == 0: TN += 1

acc = (TP + TN) / (TP + FP + TN + FN)
print(f"\nValidation Accuracy: {acc * 100:.2f}%")

print("\n[6/6] Saving model...")
torch.save(model, "./model_train/model-cc_1.pkl")
print("✓ Model saved to: ./model_train/model-cc_1.pkl")
print("\n" + "=" * 80)
print("TRAINING COMPLETE! Next step: python CNN_process.py")
print("=" * 80)