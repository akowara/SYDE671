import numpy as np
import matplotlib.pyplot as plt
import skimage.io as skio
from skimage import img_as_float, img_as_ubyte
from scipy.signal import convolve2d
import cv2
import os

def conv_color(img, kernel):
    if img.ndim == 2:
        return convolve2d(img, kernel, mode="same", boundary="symm")
    return np.stack(
        [convolve2d(img[..., c], kernel, mode="same", boundary="symm")
         for c in range(img.shape[2])],
        axis=-1,
    )

def main():
    img_filenames = ["input_images/taj.jpg","input_images/part2_1.jpg"]
    for img_filename in img_filenames:
        img = img_as_float(skio.imread(img_filename))
        name = os.path.splitext(os.path.basename(img_filename))[0]
        if img_filename == "input_images/part2_1.jpg":
            img = img[::4, ::4]

        sigma = 2
        ksize = 6 * sigma + 1
        g = cv2.getGaussianKernel(ksize, sigma)
        G = g @ g.T

        alpha = 1.5

        blurred_img = conv_color(img, G)
        high_freqs = img - blurred_img
        # sharpened_steps = blurred_img + alpha * high_freqs
        sharpened_steps = img + alpha * high_freqs

        e = np.zeros_like(G)      
        e[ksize // 2, ksize // 2] = 1
        unsharp_filter = (1 + alpha) * e - alpha * G
        sharpened_single = np.clip(conv_color(img, unsharp_filter), 0, 1)

        print("max difference:", np.abs(np.clip(sharpened_steps, 0, 1) - sharpened_single).max())
        
        high_vis = np.clip(high_freqs * 3 + 0.5, 0, 1)
        skio.imsave(f"results/part2_1_{name}_results.jpg", img_as_ubyte(sharpened_single))

        fig, ax = plt.subplots(1, 4, figsize=(20, 5))
        ax[0].imshow(img);              ax[0].set_title('Original')
        ax[1].imshow(blurred_img);          ax[1].set_title(f'Low frequencies (σ={sigma})')
        ax[2].imshow(high_vis);         ax[2].set_title('High frequencies (scaled)')
        ax[3].imshow(sharpened_single); ax[3].set_title(f'Sharpened (α={alpha})')

        for a in ax: a.axis('off')
        plt.tight_layout()
        plt.savefig(f"results/part2_1_{name}_sharpening.jpg", dpi=150)
        plt.show()

        alphas = [0.5, 1, 2, 4]
        fig, ax = plt.subplots(1, len(alphas), figsize=(20, 5))
        for a, al in zip(ax, alphas):
            f = (1 + al) * e - al * G
            a.imshow(np.clip(conv_color(img, f), 0, 1))
            a.set_title(f'α = {al}'); a.axis('off')
        plt.tight_layout()
        plt.savefig(f"results/part2_1_{name}_alphas.jpg", dpi=150)
        plt.show()

    evaluate_resharpen("input_images/skyline.jpg")


def psnr(a, b):
    return 10 * np.log10(1 / np.mean((a - b) ** 2))

def evaluate_resharpen(path, sigma=2, alphas=(1, 2, 4)):
    img = img_as_float(skio.imread(path))[::8, ::8, :3]
    ksize = 6 * sigma + 1
    g = cv2.getGaussianKernel(ksize, sigma)
    G = g @ g.T
    e = np.zeros_like(G)
    e[ksize // 2, ksize // 2] = 1

    blurred = conv_color(img, G)
    panels = [("Original (sharp)", img), (f"Blurred (σ={sigma})  PSNR {psnr(img, blurred):.1f} dB", blurred)]
    for al in alphas:
        sharp = np.clip(conv_color(blurred, (1 + al) * e - al * G), 0, 1)
        panels.append((f"Re-sharpened α={al}  PSNR {psnr(img, sharp):.1f} dB", sharp))
        print(f"alpha={al}: PSNR {psnr(img, sharp):.2f} dB")

    h, w = img.shape[:2]
    crop = (slice(int(h * 0.6), int(h * 0.85)), slice(int(w * 0.7), int(w * 0.95)))
    fig, ax = plt.subplots(2, len(panels), figsize=(5 * len(panels), 8))
    for c, (t, im) in enumerate(panels):
        ax[0, c].imshow(im);       ax[0, c].set_title(t, fontsize=10)
        ax[1, c].imshow(im[crop]); ax[1, c].set_title("zoomed crop", fontsize=9)
    for a in ax.ravel(): a.axis('off')
    plt.tight_layout()
    plt.savefig("results/part2_1_resharpen.jpg", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
    