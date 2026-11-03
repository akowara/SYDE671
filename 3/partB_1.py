import partA_2 as a2
import cv2
import numpy as np
import matplotlib.pyplot as plt
import harris as harris

def show_corners(image, cords, i, use_anms=False):

    fig, ax = plt.subplots(figsize=(8, 8))

    cords = np.asarray(cords)
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax.scatter(cords[1], cords[0], s=10, c='r', marker='o')   # xs, ys
    ax.set_title(f"{cords.shape[1]} corners {'after ANMS' if use_anms else 'before ANMS'}")
    ax.axis("off")

    plt.tight_layout()
    
    suffix = '_anms' if use_anms else ''
    plt.savefig(str(a2.RESULTS_DIR / f'harris_corners{suffix}_{i}.jpeg'))
    plt.show()
    plt.close(fig)

def anms(h, cords, n_keep=500, c_robust=0.9):
    strengths = h[cords[0], cords[1]]
    top = np.argsort(-strengths)[:10000]
    cords = cords[:, top]
    strengths = strengths[top]
    pts = cords.T

    D = harris.dist2(pts, pts)
    stronger = strengths[:, None] < c_robust * strengths[None, :]
    D[~stronger] = np.inf
    radii = np.min(D, axis=1)

    keep = np.argsort(-radii)[:n_keep]
    return cords[:, keep]

def main():
    images = a2.read_images(a2.IMAGES_DIR)
    reponses = []
    corners = []
    anms_corners = []

    for i, image in enumerate(images):
        gray_image = np.float32(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        h, cords = harris.get_harris_corners(gray_image)

        reponses.append(h)
        corners.append(cords)
        show_corners(image, cords, i)

        anms_cords = anms(h, cords)
        anms_corners.append(anms_cords)
        show_corners(image, anms_cords, i, use_anms=True)

if __name__ == "__main__":
    main()