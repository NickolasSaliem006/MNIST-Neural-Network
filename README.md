# MNIST Neural Network from Scratch

A handwritten-digit classifier built with **NumPy only**: no PyTorch, no autograd. Every part of the network, including the forward pass, the loss, and backpropagation, is written by hand. It includes an interactive app where you draw a digit with your mouse and watch the network guess it live.

**Test accuracy: 97.9%** on the 10,000-image MNIST test set.

---

## Features

### 1. Neural network written by hand
- Fully connected network: **784 → 128 (ReLU) → 10 (softmax)**
- He initialization, numerically stable softmax, cross-entropy loss
- Backpropagation derived and implemented manually, including the `P − Y` softmax + cross-entropy gradient
- Minibatch SGD with per-epoch shuffling

### 2. Best-checkpoint saving
- After every epoch, test accuracy is compared with the best result so far
- The weights are saved to `data/best_params.npz` **only when they improve**
- The best accuracy is stored inside the same file, so a new run only overwrites it if it beats **every previous run**, not just its own earlier epochs

### 3. Resume training
- Set `RESUME = True` in `train.py` to load the saved best weights instead of starting from random ones
- Resumed runs use a smaller learning rate (`0.01`) to fine-tune around the best result

### 4. Draw-your-own-digit app
- Draw a digit on a 280×280 canvas with the mouse
- The network predicts **live** while you draw, with a bar chart of all 10 probabilities
- A middle panel shows **what the network actually sees**, the 28×28 preprocessed input
- A **Clear** button (or press `c`) resets the canvas

### 5. MNIST-style preprocessing
A mouse drawing looks nothing like a scanned MNIST digit, so the app converts it the same way MNIST was built:
1. Crop to the bounding box of the ink
2. Pad to a square, so tall digits like `1` keep their shape
3. Resize to 20×20 with block averaging, which also softens the edges
4. Place in a 28×28 frame with a 4-pixel border
5. Shift so the center of mass sits at the middle, (14, 14)

Without these steps the network sees digits in positions and sizes it was never trained on, and accuracy drops sharply.

---

## Project structure

```
MNIST/
├── data.py        # loads the MNIST IDX .gz files, normalizes pixels to [0, 1]
├── model.py       # init_params, relu, softmax, forward, cross_entropy, backprop
├── train.py       # training loop, best-checkpoint saving, resume option
├── draw.py        # interactive drawing + live prediction app
└── data/          # MNIST files + best_params.npz (not tracked by git)
```

---

## Setup

Requires Python 3 with NumPy and matplotlib.

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install numpy matplotlib
```

Download the four MNIST files into `data/`:

```
train-images-idx3-ubyte.gz
train-labels-idx1-ubyte.gz
t10k-images-idx3-ubyte.gz
t10k-labels-idx1-ubyte.gz
```

---

## Usage

**Train the network**

```powershell
python train.py
```

Prints test loss and accuracy each epoch, and saves `data/best_params.npz` whenever a new best is reached.

**Draw and predict**

```powershell
python draw.py
```

Requires a trained `data/best_params.npz`, so run `train.py` first.

Tips for better predictions:
- Draw large; the preprocessing scales it down anyway
- MNIST handwriting is American-style: write `1` as a single vertical stroke and `7` without a crossbar
- If a guess is wrong, look at the middle panel; the reason is usually visible there

---

## Results

| Setting | Test accuracy |
|---|---|
| `lr = 0.01`, 20 epochs | 95.6% |
| `lr = 0.1`, 20 epochs | **97.9%** |

With `lr = 0.1`, accuracy passes 96% by epoch 4 and plateaus around 97.7–97.9% after roughly epoch 13, which is the practical ceiling for this MLP.

**Known limitation:** accuracy on mouse-drawn digits is lower than on the test set. Mouse strokes differ from scanned handwriting (distribution shift), and a fully connected network has no built-in sense of where neighboring pixels are. A convolutional network handles this much better; see the roadmap.

**Note on evaluation:** the best checkpoint is currently chosen by test accuracy, which makes the reported number slightly optimistic. A separate validation split is planned.

---

## Roadmap

- [x] **Phase 1:** NumPy MLP from scratch, ≥96% target
- [x] Best-checkpoint saving, resume training, drawing app
- [ ] **Phase 2:** gradient checking, shape audit, break-it experiments
- [ ] **Phase 3:** same network in PyTorch on GPU, ≥98%
- [ ] **Phase 4:** convolutional network, ≥99%
- [ ] **Phase 5:** GPU profiling and a small custom CUDA kernel
Longer term, this project is the foundation for building a small GPT-style language model from scratch.