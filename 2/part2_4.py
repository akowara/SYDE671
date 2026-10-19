import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_filter(img, kernel):
    return cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REFLECT)

def low_pass_filter(sigma):
    ksize = int(6 * sigma) | 1 
    g = cv2.getGaussianKernel(ksize, sigma)
    return g @ g.T

def stacks(img, num_levels=5, sigma=2):
    gaussian, laplacian = [img], []
    for i in range(num_levels):
        blurred = apply_filter(img, low_pass_filter(sigma * 2**i))  # σ doubles
        laplacian.append(gaussian[-1] - blurred)
        gaussian.append(blurred)
    laplacian.append(gaussian[-1])
    return gaussian, laplacian

def load_rgb(path, size=None, max_side=1000):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    if size is not None: 
        h, w = img.shape[:2]
        tw, th = size
        if w * th > h * tw:
            new_w = round(h * tw / th)
            img = img[:, (w - new_w) // 2:(w - new_w) // 2 + new_w]
        else:
            new_h = round(w * th / tw)
            img = img[(h - new_h) // 2:(h - new_h) // 2 + new_h]
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    elif max(img.shape[:2]) > max_side:
        scale = max_side / max(img.shape[:2])
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255

def to_gray(img):
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return np.repeat(g[..., None], 3, axis=-1)

def align_to(img_b, pts_a, pts_b, w, h):
    hb, wb = img_b.shape[:2]
    src = np.float32([(x * wb, y * hb) for x, y in pts_b])
    dst = np.float32([(x * w, y * h) for x, y in pts_a])
    M, _ = cv2.estimateAffinePartial2D(src, dst)
    return cv2.warpAffine(img_b, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

def make_mask(spec, h, w):
    if spec == "vertical":
        mask = np.zeros((h, w), np.float32)
        mask[:, :w // 2] = 1.0
    elif spec == "horizontal":
        mask = np.zeros((h, w), np.float32)
        mask[:h // 2, :] = 1.0
    else:
        mask = cv2.imread(spec, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(spec)
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 127).astype(np.float32)
    return np.repeat(mask[..., None], 3, axis=-1)

def blend(img_a, img_b, mask, num_levels=5, sigma=5):
    _, LA = stacks(img_a, num_levels, sigma)
    _, LB = stacks(img_b, num_levels, sigma)
    GM, _ = stacks(mask, num_levels, sigma)

    LA_m = [GM[i] * LA[i]       for i in range(len(LA))]
    LB_m = [(1 - GM[i]) * LB[i] for i in range(len(LB))]
    LS   = [a + b for a, b in zip(LA_m, LB_m)]
    return LA_m, LB_m, LS, np.clip(sum(LS), 0, 1)

def norm(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-8)

def show_blend(img_a, img_b, mask, result, title=""):
    fig, ax = plt.subplots(1, 4, figsize=(16, 4))
    for a, im, t in zip(ax, [img_a, img_b, mask, result], ["Image A", "Image B", "Mask (white = A)", "Blend"]):
        a.imshow(im); a.set_title(t); a.axis("off")
    ax[2].axis("on"); ax[2].set_xticks([]); ax[2].set_yticks([]) 
    fig.suptitle(title)
    plt.tight_layout()

def show_laplacian_stack(LA_m, LB_m, LS, levels=(0, 2, 4), title=""):
    fig, ax = plt.subplots(len(levels) + 1, 3, figsize=(10, 3.3 * (len(levels) + 1)))
    for r, lvl in enumerate(levels):
        ax[r, 0].imshow(norm(LA_m[lvl])); ax[r, 0].set_ylabel(f"level {lvl}")
        ax[r, 1].imshow(norm(LB_m[lvl]))
        ax[r, 2].imshow(norm(LS[lvl]))
    ax[-1, 0].imshow(np.clip(sum(LA_m), 0, 1))
    ax[-1, 1].imshow(np.clip(sum(LB_m), 0, 1))
    ax[-1, 2].imshow(np.clip(sum(LS), 0, 1))
    for c, t in enumerate(["Masked A", "Masked B", "Blend"]):
        ax[0, c].set_title(t)
    for a in ax.ravel():
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle(title)
    plt.tight_layout(rect=(0, 0, 1, 0.97)) 

def main():
    os.makedirs("results", exist_ok=True)

    blends = [
        ("input_images/burt_apple.png", "input_images/burt_orange.png", "vertical", "oraple"),
        ("input_images/dog.jpg", "input_images/lion.jpg", "vertical", "dog_lion"),
        ("input_images/rockies.jpg", "input_images/moon.jpg", "input_images/rockies_mask.png", "moon_rockies"),
        ("input_images/astronaut.jpg", "input_images/forest_night.jpg", "input_images/astronaut_mask.png", "astronaut_forest"),\
        ("input_images/dog.jpg", "input_images/moon.jpg", "input_images/dog_eye_mask.png", "moon_eye"),
    ]
    favourite = "moon_rockies" 

    align_points = {
        "dog_lion": ([(0.295, 0.387), (0.768, 0.395)], 
                     [(0.263, 0.305), (0.725, 0.308)]), 
        "moon_eye": ([(0.2988, 0.3843), (0.2988 + 0.0392, 0.3843)], 
                     [(0.5073, 0.4863), (0.5073 + 0.1311, 0.4863)]), 
    }

    for (path_a, path_b, mask_spec, base), gray in [(b, g) for g in (False, True) for b in blends]:
        name = f"{base}_gray" if gray else base
        img_a = load_rgb(path_a)
        h, w = img_a.shape[:2]
        if base in align_points:
            img_b = align_to(load_rgb(path_b), *align_points[base], w, h)
        else:
            img_b = load_rgb(path_b, size=(w, h))
        if gray:
            img_a, img_b = to_gray(img_a), to_gray(img_b)
        mask = make_mask(mask_spec, h, w)

        LA_m, LB_m, LS, result = blend(img_a, img_b, mask, num_levels=5, sigma=5)

        show_blend(img_a, img_b, mask, result, name)
        plt.savefig(f"results/part2_4_{name}_blend.jpg", dpi=150)
        out = cv2.cvtColor((result * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"results/part2_4_{name}.jpg", out)

        if base == favourite:
            show_laplacian_stack(LA_m, LB_m, LS, title=name)
            plt.savefig(f"results/part2_4_{name}_stack.jpg", dpi=150)

    plt.show()


if __name__ == "__main__":
    main()
