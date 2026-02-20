import torch
import torch.nn as nn
import numpy as np

loss_fn = nn.CrossEntropyLoss()

def get_batch(vectorized_data, seq_length, batch_size):
    n = len(vectorized_data) - 1
    idx = np.random.choice(n - seq_length, batch_size)

    input_batch = [vectorized_data[i:i+seq_length] for i in idx]
    target_batch = [vectorized_data[i+1:i+seq_length+1] for i in idx]

    x = torch.tensor(input_batch, dtype=torch.long)
    y = torch.tensor(target_batch, dtype=torch.long)

    return x, y

def compute_loss(labels, logits):
    labels = labels.view(-1)
    logits = logits.view(-1, logits.size(-1))
    return loss_fn(logits, labels)
