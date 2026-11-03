import partA_2 as a2
import partB_1 as b1
import partB_2 as b2
import cv2
import numpy as np
import matplotlib.pyplot as plt
import harris as harris

def match_features(desc1, desc2, ratio=0.5):
    D = harris.dist2(desc1, desc2)
    idx = np.argsort(D, axis=1)
    rows = np.arange(len(D))
    nn1, nn2 = idx[:, 0], idx[:, 1]
    r = D[rows, nn1] /  D[rows, nn2]
    good = r < ratio

    # back = np.argmin(D, axis=0)   
    # good = good & (back[nn1] == rows)   

    return np.stack([rows[good], nn1[good]], axis=1)

def show_matches(image1, image2, cords1, cords2, matches, name):
    h = max(image1.shape[0], image2.shape[0])
    pad = lambda im: np.pad(im, ((0, h - im.shape[0]), (0, 0), (0, 0)))
    canvas = np.hstack([pad(image1), pad(image2)])
    off = image1.shape[1]

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))

    for i, j in matches:
        y1, x1 = cords1[:, i]
        y2, x2 = cords2[:, j]
        ax.plot([x1, x2 + off], [y1, y2], '-', linewidth=0.8)
        ax.scatter([x1, x2 +off], [y1,y2], s=8, c='r')
    ax.set_title(f"{len(matches)} matches")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(str(a2.RESULTS_DIR / f'matches_{name}.jpeg'))
    plt.show()
    plt.close(fig)

def main():
    images = a2.read_images(a2.IMAGES_DIR)
    reponses = []
    corners = []
    anms_corners = []
    descriptors = []

    for i, image in enumerate(images):
        gray_image = np.float32(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        h, cords = harris.get_harris_corners(gray_image)
        anms_cords = b1.anms(h, cords)
        anms_corners.append(anms_cords)
        descriptors.append(b2.get_descriptors(gray_image, anms_cords))

    for  k in range(0, len(images) - 1, 2):
        matches = match_features(descriptors[k], descriptors[k + 1])
        show_matches(images[k], images[k + 1], anms_corners[k], anms_corners[k + 1], matches, f"{k}_{k+1}")
        

if __name__ == "__main__":
    main()