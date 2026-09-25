import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from pathlib import Path
import model as md


path = Path(__file__).parent / "data" / "best_params.npz"
data = np.load(path)
params = {k: data[k] for k in data.files if k != "acc"}


canvas = np.zeros((280, 280), dtype=np.float32)
yy, xx = np.ogrid[:280, :280]
R = 12
drawing = False



def preprocess(canvas):
    ys, xs = np.nonzero(canvas)
    if len(ys) == 0:
        return np.zeros((1, 784), dtype=np.float32)

    crop = canvas[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = crop.shape
    s = max(h, w)
    sq = np.zeros((s, s), dtype=np.float32)
    top, left = (s - h) // 2, (s - w) // 2
    sq[top:top + h, left:left + w] = crop

    idx = np.linspace(0, s - 1, 200).astype(int)
    big = sq[np.ix_(idx, idx)]
    small = big.reshape(20, 10, 20, 10).mean(axis=(1, 3))

    img = np.zeros((28, 28), dtype=np.float32)
    img[4:24, 4:24] = small

    r, c = np.indices(img.shape)
    total = img.sum()
    cy = (r * img).sum() / total
    cx = (c * img).sum() / total
    img = np.roll(img, (round(14 - cy), round(14 - cx)), axis=(0, 1))

    return img.reshape(1, 784)



fig, (ax_draw, ax_small, ax_bar) = plt.subplots(1, 3, figsize=(12, 4))
plt.subplots_adjust(bottom=0.2)

im = ax_draw.imshow(canvas, cmap="gray", vmin=0, vmax=1)
ax_draw.set_title("draw here")
ax_draw.axis("off")

im_small = ax_small.imshow(np.zeros((28, 28)), cmap="gray", vmin=0, vmax=1)
ax_small.set_title("what the network sees")
ax_small.axis("off")

bars = ax_bar.bar(range(10), np.zeros(10))
ax_bar.set_ylim(0, 1)
ax_bar.set_xticks(range(10))



def paint(event):
    if event.inaxes is not ax_draw or event.xdata is None:
        return
    mask = (xx - event.xdata) ** 2 + (yy - event.ydata) ** 2 <= R ** 2
    canvas[mask] = 1.0


def update():
    x = preprocess(canvas)
    im_small.set_data(x.reshape(28, 28))
    P, _ = md.forward(x, params)
    probs = P[0]
    for b, p in zip(bars, probs):
        b.set_height(p)
    ax_bar.set_title(f"guess: {probs.argmax()}  ({probs.max():.0%})")
    im.set_data(canvas)
    fig.canvas.draw_idle()


def clear(event):
    canvas[:] = 0
    update()



def on_press(event):
    global drawing
    drawing = True
    paint(event)
    update()


def on_move(event):
    if drawing:
        paint(event)
        update()


def on_release(event):
    global drawing
    drawing = False


def on_key(event):
    if event.key == "c":
        clear(event)


fig.canvas.mpl_connect("button_press_event", on_press)
fig.canvas.mpl_connect("motion_notify_event", on_move)
fig.canvas.mpl_connect("button_release_event", on_release)
fig.canvas.mpl_connect("key_press_event", on_key)

# ---------- clear button ----------
ax_btn = fig.add_axes([0.45, 0.05, 0.1, 0.07])
btn = Button(ax_btn, "Clear")
btn.on_clicked(clear)

plt.show()