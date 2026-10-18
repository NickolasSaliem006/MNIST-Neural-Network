from pathlib import Path
import numpy as np
import torch
from data import load_images, load_labels
import torch.nn.functional as F
import time
import model as md


device = torch.device("cuda")

# load data
X_train = load_images("data/train-images-idx3-ubyte.gz")
labels_train = load_labels("data/train-labels-idx1-ubyte.gz")
X_test = load_images("data/t10k-images-idx3-ubyte.gz")
labels_test = load_labels("data/t10k-labels-idx1-ubyte.gz")

# NumPy -> tensors, copied to the GPU once
X_train = torch.tensor(X_train, dtype=torch.float32, device=device)
y_train = torch.tensor(labels_train, dtype=torch.long, device=device)
X_test = torch.tensor(X_test, dtype=torch.float32, device=device)
y_test = torch.tensor(labels_test, dtype=torch.long, device=device)

rng = np.random.default_rng(0)
params_np = md.init_params(rng)
params = {k: torch.tensor(v, dtype=torch.float32, device=device, requires_grad=True)
          for k, v in params_np.items()}

W1 = params["W1"]
b1 = params["b1"]
W2 = params["W2"]
b2 = params["b2"]

lr= 0.07
batch = 600
epochs = 100

t0 = time.time()
for i in range(epochs):
    idx = torch.randperm(len(X_train), device=device)
    Xs = X_train[idx]
    ys = y_train[idx]

    for start in range(0, len(Xs), batch):
        Xb = Xs[start:start+batch]
        yb = ys[start:start+batch]

        Z1 = Xb @ W1 + b1
        A1 = torch.relu(Z1)
        Z2 = A1 @ W2 + b2
        loss = F.cross_entropy(Z2, yb)

        loss.backward()

        with torch.no_grad():
            for key in params:
                params[key] -= lr * params[key].grad
                params[key].grad = None

    with torch.no_grad():
        Z2 = torch.relu(X_test @ W1 + b1) @ W2 + b2
        test_loss = F.cross_entropy(Z2, y_test).item()
        acc = (Z2.argmax(dim=1) == y_test).float().mean().item()
    print(f"epoch {i+1}: test loss {test_loss:.4f}  test acc {acc:.4f}")

print("training seconds:", time.time() - t0)