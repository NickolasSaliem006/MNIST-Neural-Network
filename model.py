import numpy as np


def init_params(rng):
    W1 = rng.standard_normal((784,128))*np.sqrt(2/784)
    b1 = np.zeros(128)
    W2 = rng.standard_normal((128,10))*np.sqrt(2/128)
    b2 = np.zeros(10)

    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

def relu(Z):

    return np.maximum(0,Z)

def softmax(Z):
    Z= Z-Z.max(axis= 1, keepdims=True)
    E = np.exp(Z)
    return E/E.sum(axis=1, keepdims=True)

def forward(X,params):
    Z1 = X @ params["W1"] + params["b1"]
    A1 = relu(Z1)
    Z2 = A1 @ params["W2"] + params["b2"]
    P = softmax(Z2)
    cache = {"Z1": Z1, "A1": A1, "Z2": Z2}
    return P, cache

if __name__ == "__main__":
    from data import load_images, load_labels
    X_train = load_images("data/train-images-idx3-ubyte.gz")
    y_train = load_labels("data/train-labels-idx1-ubyte.gz")

    rng=np.random.default_rng(0)
    params = init_params(rng)

    X = X_train[:64]
    y = y_train[:64]

    P,cache = forward(X,params)

    print("Z1", cache["Z1"].shape)
    print("A1", cache["A1"].shape)
    print("Z2", cache["Z2"].shape)
    print("P ", P.shape)
    print("rows sum to 1:", np.allclose(P.sum(axis=1), 1))

    preds = np.argmax(P,axis=1)
    print("accuracy:", (preds==y).mean())

