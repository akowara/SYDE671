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
        blurred = apply_filter(img, low_pass_filter(sigma * 2**i))
        laplacian.append(gaussian[-1] - blurred)
        gaussian.append(blurred)
    laplacian.append(gaussian[-1])
    return gaussian, laplacian

def main():
    apple_img  = cv2.imread("input_images/burt_apple.png")
    orange_img = cv2.imread("input_images/burt_orange.png")
    apple_img  = cv2.cvtColor(apple_img,  cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    orange_img = cv2.cvtColor(orange_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    orange_img = cv2.resize(orange_img, (apple_img.shape[1], apple_img.shape[0]))


    mask = np.zeros_like(apple_img, dtype=np.float32)
    mask[:, :mask.shape[1] // 2, :] = 1.0

    apple_gaussian, apple_laplacian = stacks(apple_img, num_levels=5, sigma=5)
    orange_gaussian, orange_laplacian = stacks(orange_img, num_levels=5, sigma=5)
    mask_gaussian, mask_laplacian = stacks(mask, num_levels=5, sigma=5)

    LA_m = [mask_gaussian[i] * apple_laplacian[i]        for i in range(len(apple_laplacian))]
    LB_m = [(1 - mask_gaussian[i]) * orange_laplacian[i] for i in range(len(orange_laplacian))]
    LS   = [a + b for a, b in zip(LA_m, LB_m)]

    oraple = np.clip(sum(LS), 0, 1)

    def norm(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-8)

    rows = [0, 2, 4]
    fig, ax = plt.subplots(4, 3, figsize=(10, 13))
    for r, lvl in enumerate(rows):
        ax[r, 0].imshow(norm(LA_m[lvl]))
        ax[r, 1].imshow(norm(LB_m[lvl]))
        ax[r, 2].imshow(norm(LS[lvl]))
    ax[3, 0].imshow(np.clip(sum(LA_m), 0, 1))
    ax[3, 1].imshow(np.clip(sum(LB_m), 0, 1))
    ax[3, 2].imshow(oraple)
    for i, a in enumerate(ax.ravel()):
        a.set_xticks([]); a.set_yticks([])
        a.set_xlabel(f"({'abcdefghijkl'[i]})")
    for r, lvl in enumerate(rows):
        ax[r, 0].set_ylabel(f"level {lvl}")
    ax[3, 0].set_ylabel("sum")
    for c, t in enumerate(["Masked apple", "Masked orange", "Blend"]):
        ax[0, c].set_title(t)
    plt.tight_layout()
    plt.savefig("results/part2_3_oraple_stacks.jpg", dpi=150)
    plt.show()

    stack_rows = [
        ("Apple Gaussian", apple_gaussian), ("Apple Laplacian", apple_laplacian),
        ("Orange Gaussian", orange_gaussian), ("Orange Laplacian", orange_laplacian),
    ]
    fig, ax = plt.subplots(4, len(apple_gaussian), figsize=(3 * len(apple_gaussian), 9.5))
    for r, (t, stack) in enumerate(stack_rows):
        for c, im in enumerate(stack):
            ax[r, c].imshow(im if "Gaussian" in t or c == len(stack) - 1 else norm(im))
            ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
        ax[r, 0].set_ylabel(t)
    for c in range(len(apple_gaussian)):
        ax[0, c].set_title(f"level {c}")
    plt.tight_layout()
    plt.savefig("results/part2_3_gaussian_laplacian_stacks.jpg", dpi=150)
    plt.show()

    out = cv2.cvtColor((oraple * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite("results/part2_3_oraple.jpg", out)
    

if __name__ == "__main__":
    main()