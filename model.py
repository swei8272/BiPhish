import torch
from torch import nn
from torch.nn import functional as F


class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_class):
        super(CNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.conv1 = nn.Conv1d(embedding_dim, 64, 3)
        self.conv2 = nn.Conv1d(embedding_dim, 64, 5)
        # 64 filters * 2 branches = 128 features
        self.linear = nn.Linear(128, num_class)

    def forward(self, x):
        x = self.embedding(x)

        # Permute for Conv1d: (Batch, Seq_Len, Embed) -> (Batch, Embed, Seq_Len)
        x1 = F.relu(self.conv1(x.permute(0, 2, 1)))
        x2 = F.relu(self.conv2(x.permute(0, 2, 1)))

        pool1 = F.max_pool1d(x1, kernel_size=x1.shape[2])
        pool2 = F.max_pool1d(x2, kernel_size=x2.shape[2])

        x1 = pool1.squeeze(dim=2)
        x2 = pool2.squeeze(dim=2)

        # Feature concatenation
        x = torch.cat([x1, x2], dim=1)
        self.features = x  # Save for feature extraction

        x = self.linear(x)
        out = F.log_softmax(x, dim=1)
        return out