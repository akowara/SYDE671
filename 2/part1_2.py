import os
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as skio
from skimage import img_as_float
from skimage.color import rgb2gray
from scipy.signal import convolve2d

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

def main():
    img_filename = "/Users/arthurkowara/Documents/School/Waterloo/F26/SYDE671/SYDE671/2/input_images/thecameraman.jpg"
    img = img_as_float(skio.imread(img_filename))

    if img.ndim == 3:
        img = rgb2gray(img[..., :3])  

    Dx = np.array([[1, -1]])
    Dy = np.array([[1], [-1]])

    dx = convolve2d(img, Dx, mode="same", boundary="fill", fillvalue=0)
    dy = convolve2d(img, Dy, mode="same", boundary="fill", fillvalue=0)

    gradient_magnitude = np.sqrt(dx**2 + dy**2)

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(dx, cmap='gray'); ax[0].set_title('∂I/∂x')
    ax[1].imshow(dy, cmap='gray'); ax[1].set_title('∂I/∂y')
    ax[2].imshow(gradient_magnitude, cmap='gray')
    ax[2].set_title('Gradient magnitude')
    for a in ax: a.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "part1_2_gradients.jpg"), dpi=150)
    plt.show()

    thresholds = [0.05, 0.075, 0.1, 0.125,0.2, 0.3,]
    fig, ax = plt.subplots(2, 3, figsize=(15, 10))
    for a, t in zip(ax.ravel(), thresholds):
        a.imshow(gradient_magnitude > t, cmap='gray')
        a.set_title(f'threshold = {t}')
        a.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "part1_2_thresholds.jpg"), dpi=150)
    plt.show()

    edges = (gradient_magnitude > 0.05).astype(np.uint8) * 255
    skio.imsave(os.path.join(RESULTS_DIR, "part1_2_results.jpg"), edges)

if __name__ == "__main__":
    main()