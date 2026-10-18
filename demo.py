"""Interactive demo: image compression with SVD (matrix compression).

Run:   python demo.py

- Drag the slider (or press the arrow keys: left/right = 1 step, up/down = 10 steps)
  to choose k, the number of SVD "layers" we keep.
- Click an animal name in the bottom-left corner to switch images.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import RadioButtons, Slider
from PIL import Image

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
IMAGE_DIR = Path(__file__).parent / "images"
IMAGES = {"Shark": "shark.jpg", "Deer": "deer.jpg", "Leopard": "leopard.jpg"}
K_MAX = 300    # largest k on the slider
K_START = 20   # k shown when the demo opens


# ---------------------------------------------------------------------------
# The math (the core of the project)
# ---------------------------------------------------------------------------
def load(filename):
    """Load an image and compute the SVD of each colour channel.

    A colour image is 3 stacked matrices (R, G, B), each of size m x n.
    Each entry is the brightness of one pixel, from 0 to 255.
    """
    # Turn the image into a matrix of numbers with shape (m, n, 3)
    img = np.asarray(Image.open(IMAGE_DIR / filename).convert("RGB"), dtype=float)

    # SVD of each colour:  A = U  S  V^T
    #   U  = column vectors (m x r)
    #   S  = singular values σ1 >= σ2 >= ..., how important each layer is
    #   Vt = row vectors (r x n)
    # full_matrices=False gives the thin SVD, which saves memory
    svds = [np.linalg.svd(img[:, :, c], full_matrices=False) for c in range(3)]

    # The SVD is computed once at startup; moving the slider only slices
    # the first k layers, so it stays fast
    return img, svds


def compress(svds, k):
    """Keep only the first k layers of each colour:  A_k = U_k  S_k  V_k^T

    This equals  σ1·u1·v1^T + σ2·u2·v2^T + ... + σk·uk·vk^T
    By the Eckart–Young theorem, this is the best possible rank-k approximation.
    """
    channels = [
        U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]  # first k columns of U, k values of S, k rows of Vt
        for U, S, Vt in svds                   # one colour at a time: R, G, B
    ]
    # Stack the 3 colours back into an image and clip to 0-255
    # (the approximation can go slightly outside that range)
    return np.clip(np.stack(channels, axis=2), 0, 255).astype(np.uint8)


def storage(img, k):
    """Fraction of storage used compared with the original image.

    The original stores m*n numbers per colour.
    The compressed form stores U_k (m*k) + S_k (k) + V_k (n*k) = k(m + n + 1) numbers per colour.
    """
    m, n, _ = img.shape
    return k * (m + n + 1) / (m * n)


def psnr(img, approx):
    """Image quality in dB: higher means closer to the original.

    PSNR = 10 log10(255^2 / MSE), where MSE = mean of (pixel difference)^2
    """
    mse = np.mean((img - approx) ** 2)
    return 10 * np.log10(255 ** 2 / mse)


def quality_label(db):
    """Turn a dB value into a word the audience can understand."""
    if db < 22:
        return "poor"
    if db < 27:
        return "fair"
    if db < 32:
        return "good"
    if db < 40:
        return "very good"
    return "almost identical"


# ---------------------------------------------------------------------------
# The demo window
# ---------------------------------------------------------------------------
def build_demo():
    # Precompute the SVD of every image so switching images is instant
    print("Computing SVD for each image...")
    data = {name: load(filename) for name, filename in IMAGES.items()}
    state = {"name": "Shark"}  # image currently shown

    # --- Layout: [left, bottom, width, height] as fractions of the window ---
    fig = plt.figure(figsize=(14, 8.5))
    fig.suptitle("Matrix Compression with SVD", fontsize=20, fontweight="bold")

    ax_orig = fig.add_axes([0.03, 0.40, 0.45, 0.50])    # original image (top left)
    ax_comp = fig.add_axes([0.52, 0.40, 0.45, 0.50])    # compressed image (top right)
    ax_slider = fig.add_axes([0.20, 0.31, 0.62, 0.03])  # k slider
    ax_radio = fig.add_axes([0.03, 0.05, 0.13, 0.20])   # image picker (bottom left)
    ax_sv = fig.add_axes([0.26, 0.07, 0.71, 0.18])      # singular value plot (bottom right)
    info = fig.text(0.5, 0.365, "", ha="center", fontsize=15)  # storage + quality text

    # --- Original and compressed images ---
    img, svds = data[state["name"]]
    orig_view = ax_orig.imshow(img.astype(np.uint8))
    comp_view = ax_comp.imshow(compress(svds, K_START))
    ax_orig.set_title("Original", fontsize=15)
    for ax in (ax_orig, ax_comp):
        ax.axis("off")  # hide the axis ticks around the images

    # --- Slider and image picker ---
    slider = Slider(ax_slider, "k (layers kept)", 1, K_MAX, valinit=K_START, valstep=1)
    slider.vline.set_visible(False)  # hide the red tick marking the starting value
    radio = RadioButtons(ax_radio, list(IMAGES))
    ax_radio.set_title("Image", fontsize=13)

    # --- Singular value plot: shows the first layers matter far more than later ones ---
    ax_sv.set_yscale("log")  # log scale, because σ1 is hundreds of times larger than the last ones
    ax_sv.set_xlim(1, K_MAX)
    ax_sv.set_xlabel("layer i")
    ax_sv.set_ylabel("importance σᵢ")
    ax_sv.set_title("Each layer is less important than the one before", fontsize=12)
    sv_line, = ax_sv.plot([], [], color="#1565c0", lw=2)          # the σi curve
    cut_line = ax_sv.axvline(K_START, color="#c62828", lw=2)       # red line = cut-off at k
    kept_area = [None]                                             # shaded area = layers kept

    def redraw(_=None):
        """Recompute and redraw whenever k or the image changes."""
        img, svds = data[state["name"]]
        k = int(slider.val)

        # 1) Compress with k layers and show the result
        approx = compress(svds, k)
        comp_view.set_data(approx)
        ax_comp.set_title(f"Compressed (k = {k})", fontsize=15)

        # 2) Update the storage and quality numbers
        db = psnr(img, approx)
        info.set_text(f"Storage used: {storage(img, k):.1%} of the original    ·    "
                      f"Quality: {db:.1f} dB ({quality_label(db)})")

        # 3) Update the σi plot (average of the 3 colours) and shade the kept part
        S = np.mean([s for _, s, _ in svds], axis=0)
        i = np.arange(1, len(S) + 1)
        floor = S[K_MAX - 1] * 0.8  # bottom edge of the plot
        sv_line.set_data(i, S)
        ax_sv.set_ylim(floor, S[0] * 1.5)
        cut_line.set_xdata([k, k])
        if kept_area[0] is not None:
            kept_area[0].remove()  # remove the old shading before drawing the new one
        kept_area[0] = ax_sv.fill_between(i[:k], S[:k], floor, color="#1565c0", alpha=0.2)

        fig.canvas.draw_idle()  # ask the window to redraw

    def change_image(name):
        """Called when a new image is picked."""
        state["name"] = name
        img = data[name][0]
        m, n, _ = img.shape
        orig_view.set_data(img.astype(np.uint8))
        # The images have different heights, so resize the frames to fit the new one
        for view, ax in ((orig_view, ax_orig), (comp_view, ax_comp)):
            view.set_extent((-0.5, n - 0.5, m - 0.5, -0.5))
            ax.set_xlim(-0.5, n - 0.5)
            ax.set_ylim(m - 0.5, -0.5)
        redraw()

    def on_key(event):
        """Arrow keys: left/right change k by 1, up/down by 10 (handy while presenting)."""
        step = {"right": 1, "left": -1, "up": 10, "down": -10}.get(event.key)
        if step:
            slider.set_val(min(max(slider.val + step, 1), K_MAX))

    # Connect events: slider -> redraw, image click -> change_image, key press -> on_key
    slider.on_changed(redraw)
    radio.on_clicked(change_image)
    fig.canvas.mpl_connect("key_press_event", on_key)
    redraw()  # first draw
    return fig, (slider, radio)


if __name__ == "__main__":
    fig, widgets = build_demo()  # keep a reference to the widgets, or the slider stops working
    plt.show()
