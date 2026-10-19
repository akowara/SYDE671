import time
import numpy as np
from scipy.signal import convolve2d
import skimage.io as skio
from skimage.color import rgb2gray
from skimage import img_as_float
import matplotlib.pyplot as plt


def zero_pad(img, kH, kW):
    pad_top = (kH - 1) // 2
    pad_bottom = kH - 1 - pad_top
    pad_left = (kW - 1) // 2
    pad_right = kW - 1 - pad_left
    H, W = img.shape
    padded = np.zeros((H + pad_top + pad_bottom, W + pad_left + pad_right))
    padded[pad_top:pad_top + H, pad_left:pad_left + W] = img
    return padded


def conv_four_loops(img, kernel):
    kernel = np.flip(kernel)
    kH, kW = kernel.shape
    H, W = img.shape
    padded = zero_pad(img, kH, kW)
    output = np.zeros((H, W))

    for i in range(H):
        for j in range(W):
            total = 0.0
            for m in range(kH):
                for n in range(kW):
                    total += padded[i + m, j + n] * kernel[m, n]
            output[i, j] = total
    return output

def conv_two_loops(img, kernel):
    kernel = np.flip(kernel)
    kH, kW = kernel.shape
    H, W = img.shape
    padded = zero_pad(img, kH, kW)
    output = np.zeros((H, W))

    for i in range(H):
        for j in range(W):
            output[i, j] = np.sum(padded[i:i + kH, j:j + kW] * kernel)
    return output

box = np.ones((9, 9)) / 81.0
Dx = np.array([[1, 0, -1]])
Dy = np.array([[1], [0], [-1]])


def compare_with_scipy(img, kernel, name):
    t0 = time.time(); out4 = conv_four_loops(img, kernel); t4 = time.time() - t0
    t0 = time.time(); out2 = conv_two_loops(img, kernel);  t2 = time.time() - t0
    t0 = time.time(); ref = convolve2d(img, kernel, mode="same", boundary="fill", fillvalue=0); ts = time.time() - t0
    print(f"{name}: four loops {t4:.2f}s | two loops {t2:.2f}s | scipy {ts:.4f}s")
    print(f"  max diff four-loop vs scipy: {np.abs(out4 - ref).max():.2e}")
    print(f"  max diff two-loop  vs scipy: {np.abs(out2 - ref).max():.2e}")
    return out2


def main():
    img = skio.imread("input_images/IMG_3234.jpg")
    img = np.rot90(img, k=-1)
    if img.ndim == 3:
        img = rgb2gray(img[..., :3])
    img = img_as_float(img)

    img = img[::4, ::4]

    blurred = compare_with_scipy(img, box, "9x9 box")
    gx = compare_with_scipy(img, Dx, "Dx")
    gy = compare_with_scipy(img, Dy, "Dy")

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, im, title in zip(axes, [img, blurred, gx, gy],
                             ["Original", "9x9 box filter", "Dx", "Dy"]):
        ax.imshow(im, cmap="gray")
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("results/part1_1_results.jpg", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()