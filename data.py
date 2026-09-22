import gzip
import numpy as np
import matplotlib .pyplot as plt

def load_images(path):
    with gzip.open(path, "rb") as f:
        buf= f.read()
        pixels = np.frombuffer(buf, dtype=np.uint8, offset=16)
        X = pixels.reshape(-1,784)
        return X.astype(np.float32) / 255.0

def load_labels(path):
    with gzip.open(path, "rb") as f:
        buf = f.read()
        return np.frombuffer(buf, dtype=np.uint8, offset=8)

if __name__ == "__main__":
    X_train = load_images("data/train-images-idx3-ubyte.gz")
    y_train = load_labels("data/train-labels-idx1-ubyte.gz")
    X_test  = load_images("data/t10k-images-idx3-ubyte.gz")
    y_test  = load_labels("data/t10k-labels-idx1-ubyte.gz")

    for name, X in [("X_train", X_train), ("X_test", X_test)]:
        print(name, X.shape, X.dtype, X.min(), X.max())
    print("first 10 labels:", y_train[:10])

    plt.imshow(X_train[0].reshape(28, 28), cmap="gray")
    plt.title(f"label: {y_train[0]}")
    plt.show()