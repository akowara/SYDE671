import os

import cv2
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
from part1_3 import my_arctan, my_arctan2

def low_pass_filter(sigma):
    ksize = int(6 * sigma + 1)     
    g = cv2.getGaussianKernel(ksize, sigma)
    return g @ g.T

def high_pass_filter(sigma):
    G = low_pass_filter(sigma)
    e = np.zeros_like(G)
    e[G.shape[0] // 2, G.shape[1] // 2] = 1
    return e - G

def apply_filter(img, kernel):
    return cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REFLECT)

def get_user_input(img, title):
    plt.figure()
    plt.imshow(img, cmap="gray")
    plt.title(title)
    pts = np.array(plt.ginput(2, timeout=0))
    plt.close()
    return pts[:, 0], pts[:, 1]

def pad_translate(img, cx, cy):
    h, w = img.shape[:2]
    tx = round((w / 2 - cx) * 2)
    ty = round((h / 2 - cy) * 2)
    extra = [(0, 0)] * (img.ndim - 2)
    img = np.pad(img, [(0, 0), (tx, 0) if tx > 0 else (0, -tx)] + extra)
    img = np.pad(img, [(ty, 0) if ty > 0 else (0, -ty), (0, 0)] + extra)
    return img

def rotate_loose(img, angle_deg):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    new_w = int(np.ceil(h * sin + w * cos))
    new_h = int(np.ceil(h * cos + w * sin))
    M[0, 2] += new_w / 2 - w / 2
    M[1, 2] += new_h / 2 - h / 2
    return cv2.warpAffine(img, M, (new_w, new_h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)

def center_crop(img, h, w):
    H, W = img.shape[:2]
    top, left = int(np.ceil((H - h) / 2)), int(np.ceil((W - w) / 2))
    return img[top:top + h, left:left + w]

def largest_rectangle(mask):
    H, W = mask.shape
    heights = np.zeros(W + 1, dtype=int)
    best, box = 0, (0, 0, 0, 0)
    for r in range(H):
        heights[:W] = np.where(mask[r], heights[:W] + 1, 0)
        stack = []
        for c in range(W + 1):
            start = c
            while stack and stack[-1][1] >= heights[c]:
                start, h = stack.pop()
                if h * (c - start) > best:
                    best, box = h * (c - start), (r - h + 1, r + 1, start, c)
            stack.append((start, heights[c]))
    top, bottom, left, right = box
    return np.s_[top:bottom, left:right]

def crop_to_overlap(img1, mask1, img2, mask2):
    box = largest_rectangle((mask1 > 0.99) & (mask2 > 0.99))
    return img1[box], img2[box]

def align_images(img1, img2):
    x1, y1 = get_user_input(img1, "Image 1: click 2 points")
    x2, y2 = get_user_input(img2, "Image 2: click the same 2 points, same order")

    mask1 = np.ones(img1.shape[:2], np.float32)
    mask2 = np.ones(img2.shape[:2], np.float32)

    img1, mask1 = [pad_translate(a, x1.mean(), y1.mean()) for a in (img1, mask1)]
    img2, mask2 = [pad_translate(a, x2.mean(), y2.mean()) for a in (img2, mask2)]

    len1 = np.hypot(x1[1] - x1[0], y1[1] - y1[0])
    len2 = np.hypot(x2[1] - x2[0], y2[1] - y2[0])
    dscale = len2 / len1
    if dscale < 1:
        img1, mask1 = [cv2.resize(a, None, fx=dscale, fy=dscale, interpolation=cv2.INTER_LINEAR) for a in (img1, mask1)]
    else:
        img2, mask2 = [cv2.resize(a, None, fx=1 / dscale, fy=1 / dscale, interpolation=cv2.INTER_LINEAR) for a in (img2, mask2)]

    theta1 = my_arctan2(-(y1[1] - y1[0]), x1[1] - x1[0])
    theta2 = my_arctan2(-(y2[1] - y2[0]), x2[1] - x2[0])

    img2, mask2 = [rotate_loose(a, np.degrees(theta1 - theta2)) for a in (img2, mask2)]

    h = min(img1.shape[0], img2.shape[0])
    w = min(img1.shape[1], img2.shape[1])
    img1, mask1, img2, mask2 = [center_crop(a, h, w) for a in (img1, mask1, img2, mask2)]

    return crop_to_overlap(img1, mask1, img2, mask2)


def make_hybrid(img_low, img_high, low_sigma, high_sigma, high_gain=1):
    # high_gain > 1 makes the high-frequency image stand out more up close
    low = apply_filter(img_low, low_pass_filter(low_sigma))
    high = apply_filter(img_high, high_pass_filter(high_sigma)) * high_gain
    hybrid = np.clip(low + high, 0, 1)
    return low, high, hybrid

def log_fft(img):
    return np.log(np.abs(np.fft.fftshift(np.fft.fft2(img))) + 1e-8)

def show_frequency_analysis(img1, img2, low, high, hybrid, title=""):
    images = [img1, img2, low, high + 0.5, hybrid]   # +0.5 so the high-pass is visible
    spectra = [img1, img2, low, high, hybrid]        # FFT of the real values
    names = ["Input 1 (low)", "Input 2 (high)", "Low-passed", "High-passed", "Hybrid"]

    fig, ax = plt.subplots(2, 5, figsize=(20, 8))
    for i in range(5):
        ax[0, i].imshow(images[i], cmap="gray", vmin=0, vmax=1)
        ax[0, i].set_title(names[i])
        ax[1, i].imshow(log_fft(spectra[i]), cmap="viridis")
        ax[1, i].set_title(f"FFT: {names[i]}")
    for a in ax.ravel():
        a.axis("off")
    fig.suptitle(title)
    plt.tight_layout()


def show_result(img1, img2, hybrid, title=""):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    for a, im, t in zip(ax, [img1, img2, hybrid], ["Input 1 (low)", "Input 2 (high)", "Hybrid"]):
        a.imshow(im, cmap="gray", vmin=0, vmax=1)
        a.set_title(t)
        a.axis("off")
    fig.suptitle(title)
    plt.tight_layout()

def pyramids(img, num_levels=5, sigma=2):
    G = low_pass_filter(sigma)
    gaussian, laplacian = [], []
    current = img
    for _ in range(num_levels):
        blurred = apply_filter(current, G)
        gaussian.append(current)
        laplacian.append(current - blurred)
        current = blurred[::2, ::2]
    return gaussian, laplacian

def show_pyramids(img, num_levels=5):
    gaussian, laplacian = pyramids(img, num_levels)
    fig, ax = plt.subplots(2, num_levels, figsize=(4 * num_levels, 8))
    for i in range(num_levels):
        ax[0, i].imshow(gaussian[i], cmap="gray"); ax[0, i].set_title(f"Gaussian {i}")
        ax[1, i].imshow(laplacian[i], cmap="gray"); ax[1, i].set_title(f"Laplacian {i}")
    for a in ax.ravel():
        a.axis("off")
    plt.tight_layout()

def load_color(path, scale=1):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    if scale != 1:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255

def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def to_rgb(gray):
    return np.repeat(gray[..., None], 3, axis=-1)

def color_variants(color1, color2, low_sigma, high_sigma, high_gain=1):
    gray1, gray2 = to_gray(color1), to_gray(color2)
    inputs = {
        "Grayscale":       (to_rgb(gray1), to_rgb(gray2)),
        "Color low only":  (color1,        to_rgb(gray2)),
        "Color high only": (to_rgb(gray1), color2),
        "Color both":      (color1,        color2),
    }
    return {k: make_hybrid(a, b, low_sigma, high_sigma, high_gain)[2] for k, (a, b) in inputs.items()}

def show_color_comparison(hybrids, title=""):
    fig, ax = plt.subplots(2, len(hybrids), figsize=(5 * len(hybrids), 8),
                           gridspec_kw={"height_ratios": [4, 1]})
    for i, (name, hybrid) in enumerate(hybrids.items()):
        ax[0, i].imshow(hybrid); ax[0, i].set_title(name)
        small = cv2.resize(hybrid, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_AREA)
        ax[1, i].imshow(small, interpolation="nearest")
        ax[1, i].set_title("from far away", fontsize=9)
    for a in ax.ravel():
        a.axis("off")
    fig.suptitle(title)
    plt.tight_layout()

def show_alignment(orig1, orig2, aligned1, aligned2, title=""):
    images = [orig1, orig2, aligned1, aligned2, (aligned1 + aligned2) / 2]
    titles = ["Input 1 (low) original", "Input 2 (high) original",
              "Input 1 aligned", "Input 2 aligned", "Aligned overlay"]
    fig, ax = plt.subplots(1, 5, figsize=(20, 5))
    for a, im, t in zip(ax, images, titles):
        a.imshow(im); a.set_title(t); a.axis("off")
    fig.suptitle(title)
    plt.tight_layout()

def main():
    os.makedirs("results", exist_ok=True)

    pairs = [
        ("input_images/DerekPicture.jpg", "input_images/nutmeg.jpg", 10, 5, 1, 1, "derek_nutmeg"),
        ("input_images/part2_2_1.jpg", "input_images/part2_2_2.jpg", 10, 5, 1, 1, "bear"),
        ("input_images/part2_2_straight.jpg","input_images/part2_2_happy.jpg", 15, 6, 1, 0.30, "happy_straight"),
    ]
    favourite = "happy_straight"

    for low_path, high_path, low_sigma, high_sigma, high_gain, scale, name in pairs:
        orig1, orig2 = load_color(low_path, scale), load_color(high_path, scale)
        color1, color2 = align_images(orig1, orig2)
        img1, img2 = to_gray(color1), to_gray(color2)
        low, high, hybrid = make_hybrid(img1, img2, low_sigma, high_sigma, high_gain)

        show_result(img1, img2, hybrid, f"{name}  (low ={low_sigma}, high ={high_sigma}, gain ={high_gain})")
        plt.savefig(f"results/part2_2_{name}_result.jpg", dpi=150)
        if name == favourite:
            show_alignment(orig1, orig2, color1, color2, f"{name}: alignment")
            plt.savefig(f"results/part2_2_{name}_alignment.jpg", dpi=150)
            show_frequency_analysis(img1, img2, low, high, hybrid, name)
            plt.savefig(f"results/part2_2_{name}_frequency.jpg", dpi=150)
            show_pyramids(hybrid)
            plt.savefig(f"results/part2_2_{name}_pyramids.jpg", dpi=150)

        cv2.imwrite(f"results/part2_2_{name}_hybrid.jpg", (hybrid * 255).astype(np.uint8))

        hybrids = color_variants(color1, color2, low_sigma, high_sigma, high_gain)
        show_color_comparison(hybrids, f"{name}: where to use color")
        plt.savefig(f"results/part2_2_{name}_color.jpg", dpi=150)
        for variant, h in hybrids.items():
            out = cv2.cvtColor((h * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"results/part2_2_{name}_{variant.lower().replace(' ', '_')}.jpg", out)

    plt.show()


if __name__ == "__main__":
    main()