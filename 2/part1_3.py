import numpy as np
import matplotlib.pyplot as plt
import skimage.io as skio
from skimage import img_as_float
from skimage.color import rgb2gray
from scipy.signal import convolve2d
import cv2
from matplotlib.colors import hsv_to_rgb

def my_arctan(z, n_terms=15):
    z = z / (1 + np.sqrt(1 + z**2))
    result = np.zeros_like(z)
    term = z.copy()
    for k in range(n_terms):
        result += term / (2 * k + 1)
        term *= -z**2
    return 2 * result

def my_arctan2(y, x):
    ax, ay = np.abs(x), np.abs(y)
    swap = ay > ax
    num = np.where(swap, ax, ay)
    den = np.where(swap, ay, ax)
    ratio = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    a = my_arctan(ratio) 
    a = np.where(swap, np.pi / 2 - a, a) 
    a = np.where(x < 0, np.pi - a, a) 
    return np.where(y < 0, -a, a) 

def main():
    img_filename = "input_images/thecameraman.jpg"
    img = img_as_float(skio.imread(img_filename))

    if img.ndim == 3:
        img = rgb2gray(img[..., :3])  

    Dx = np.array([[1, -1]])
    Dy = np.array([[1], [-1]])

    sigma = 2
    ksize = 6 * sigma + 1
    g = cv2.getGaussianKernel(ksize, sigma)
    G = g @ g.T

    # Method 1
    blurred_img = convolve2d(img, G, mode="same", boundary="fill", fillvalue=0)

    blurred_dx = convolve2d(blurred_img, Dx, mode="same", boundary="fill", fillvalue=0)
    blurred_dy = convolve2d(blurred_img, Dy, mode="same", boundary="fill", fillvalue=0)
    blurred_gradient_magnitude = np.sqrt(blurred_dx**2 + blurred_dy**2)

    # Method 2: derivative of Gaussian filters (one convolution)
    DoG_x = convolve2d(G, Dx)
    DoG_y = convolve2d(G, Dy)
    DoG_dx = convolve2d(img, DoG_x, mode="same", boundary="fill", fillvalue=0)
    DoG_dy = convolve2d(img, DoG_y, mode="same", boundary="fill", fillvalue=0)
    DoG_gradient_magnitude = np.sqrt(DoG_dx**2 + DoG_dy**2)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(img, cmap='gray');         ax[0].set_title('Original')
    ax[1].imshow(blurred_img, cmap='gray'); ax[1].set_title(f'Blurred (σ={sigma})')
    for a in ax: a.axis('off')
    plt.tight_layout(); plt.show()

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].imshow(DoG_x, cmap='gray'); ax[0].set_title('DoG_x filter')
    ax[1].imshow(DoG_y, cmap='gray'); ax[1].set_title('DoG_y filter')
    for a in ax: a.axis('off')
    plt.tight_layout(); plt.show()

    rows = [
        ('Blur then D', blurred_dx, blurred_dy, blurred_gradient_magnitude),
        ('DoG',         DoG_dx,     DoG_dy,     DoG_gradient_magnitude),
    ]
    fig, ax = plt.subplots(2, 3, figsize=(15, 10))
    for r, (name, dx, dy, mag) in enumerate(rows):
        ax[r, 0].imshow(dx,  cmap='gray'); ax[r, 0].set_title(f'{name}: ∂I/∂x')
        ax[r, 1].imshow(dy,  cmap='gray'); ax[r, 1].set_title(f'{name}: ∂I/∂y')
        ax[r, 2].imshow(mag, cmap='gray'); ax[r, 2].set_title(f'{name}: gradient magnitude')
    for a in ax.ravel(): a.axis('off')
    plt.tight_layout(); plt.show()

    diff = np.abs(blurred_gradient_magnitude - DoG_gradient_magnitude)
    m = ksize 
    print(f"Max difference (whole image):    {diff.max():.2e}")
    print(f"Max difference (interior only):  {diff[m:-m, m:-m].max():.2e}")

    plt.figure(figsize=(5, 5))
    plt.imshow(diff, cmap='gray'); plt.title('|Method 1 − Method 2|')
    plt.colorbar(fraction=0.046); plt.axis('off')
    plt.tight_layout(); plt.show()

    thresholds = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12]
    fig, ax = plt.subplots(2, 3, figsize=(15, 10))
    for a, t in zip(ax.ravel(), thresholds):
        a.imshow(DoG_gradient_magnitude > t, cmap='gray')
        a.set_title(f'threshold = {t}')
        a.axis('off')
    plt.tight_layout(); plt.show()

    theta = my_arctan2(DoG_dy, DoG_dx)
    print(f"Max |my_arctan2 - np.arctan2| (check only): {np.abs(theta - np.arctan2(DoG_dy, DoG_dx)).max():.2e}")

    hue = (theta + np.pi) / (2 * np.pi)                     # map to [0, 1]
    mag = DoG_gradient_magnitude
    val = np.clip(mag / np.percentile(mag, 99), 0, 1)       # normalize, ignore a few outliers
    hsv = np.stack([hue, np.ones_like(hue), val], axis=-1)
    rgb = hsv_to_rgb(hsv)

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    im = ax[0].imshow(theta, cmap='hsv', vmin=-np.pi, vmax=np.pi)
    ax[0].set_title('Orientation (hue only)')
    fig.colorbar(im, ax=ax[0], fraction=0.046, label='radians')
    ax[1].imshow(rgb)
    ax[1].set_title('Orientation (hue) weighted by magnitude (value)')
    for a in ax: a.axis('off')
    plt.tight_layout(); plt.show()

if __name__ == "__main__":
    main()