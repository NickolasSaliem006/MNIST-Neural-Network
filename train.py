import numpy as np
from data import load_images, load_labels
import model as md
import os
from pathlib import Path


if __name__ == "__main__":
    X_train = load_images("data/train-images-idx3-ubyte.gz")
    y_train = load_labels("data/train-labels-idx1-ubyte.gz")
    X_test = load_images("data/t10k-images-idx3-ubyte.gz")
    y_test = load_labels("data/t10k-labels-idx1-ubyte.gz")


    path = Path(__file__).parent / "data" / "best_params.npz"
    if path.exists():
        best_acc = float(np.load(path)["acc"])         
    else:
        best_acc = 0.0

    rng=np.random.default_rng(0)
    params = md.init_params(rng)
    print(f"best so far: {best_acc:.4f}")
    lr= 0.1
    batch = 64
    epochs = 20

    


    for i in range(epochs):
        idx = np.random.permutation(len(X_train))
        Xs = X_train[idx]
        ys = y_train[idx]

        for start in range(0, len(Xs), batch):
            Xb = Xs[start:start+batch]
            yb = ys[start:start+batch]
            P,cache = md.forward(Xb, params)

            N= len(yb)
        

            Y = np.zeros_like(P)
            Y[np.arange(len(yb)), yb] = 1
            grads= md.backprop(P, Y, N, cache["A1"] ,cache["Z1"] , params, Xb)

            for key in params:
                params[key] -= lr * grads[key]


        P_test, cache_test = md.forward(X_test, params)
        loss = md.cross_entropy(P_test, y_test) 
        preds = np.argmax(P_test,axis=1)
        acc=  (preds==y_test).mean()
        if acc > best_acc:
            best_acc = acc
            np.savez(path, **params, acc=acc)
            print(f"  new best {acc:.4f} at epoch {epochs}, saved")
        print(f"epoch {i+1}: test loss {loss:.4f}  test acc {acc:.4f}")


    
