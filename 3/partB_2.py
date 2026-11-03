import partA_2 as a2
import partB_1 as b1
import cv2
import numpy as np
import matplotlib.pyplot as plt
import harris as harris

def get_descriptors(image, cords):
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=2)
    decs = []
    for y, x in cords.T:
        patch = blurred[y-20:y+20, x-20:x+20]
        patch = cv2.resize(patch, (8, 8), interpolation=cv2.INTER_AREA)
        patch = (patch - patch.mean()) / (patch.std() + 1e-10)
        decs.append(patch.flatten())

    return np.array(decs)

def show_descriptors(gray, cords, descs, i, n=6):
    picks = np.linspace(0, cords.shape[1] - 1, n).astype(int)
    fig, axes = plt.subplots(2, n, figsize=(2 * n, 4.5))

    for col, k in enumerate(picks):
        y, x = cords[:, k]
        axes[0, col].imshow(gray[y-20:y+20, x-20:x+20], cmap='gray')
        axes[0, col].set_title(f"({x}, {y})", fontsize=8)
        axes[1, col].imshow(descs[k].reshape(8, 8), cmap='gray', interpolation='nearest')
        axes[0, col].axis('off')
        axes[1, col].axis('off')

    fig.suptitle("Top: 40x40 window   Bottom: 8x8 normalized descriptor")
    plt.tight_layout()
    plt.savefig(str(a2.RESULTS_DIR / f'descriptors_{i}.jpeg'))
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
        
        show_descriptors(gray_image, anms_cords, get_descriptors(gray_image, anms_cords), i)


if __name__ == "__main__":
    main()