import torch
import torch.optim as optim
from model import LSTMModel
from utils import get_batch, compute_loss
import numpy as np

def train(vectorized_data, vocab_size, params):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LSTMModel(
        vocab_size,
        params["embedding_dim"],
        params["hidden_size"]
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"])

    for iteration in range(params["num_training_iterations"]):

        x, y = get_batch(
            vectorized_data,
            params["seq_length"],
            params["batch_size"]
        )

        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        logits, _ = model(x)
        loss = compute_loss(y, logits)

        loss.backward()
        optimizer.step()

        if iteration % 100 == 0:
            print(f"Iteration {iteration} | Loss: {loss.item():.4f}")

    return model
