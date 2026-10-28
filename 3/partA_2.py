import cv2
from pathlib import Path
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
IMAGES_DIR = BASE_DIR / 'images'
RESULTS_DIR = BASE_DIR / 'results'
RESULTS_DIR.mkdir(exist_ok=True)


def read_images(folderpath=IMAGES_DIR):
    image_paths = sorted(p for p in Path(folderpath).iterdir() if p.suffix.lower() in ('.jpg', '.jpeg'))
    return [cv2.resize(cv2.imread(str(image_path)), None, fx=0.2, fy=0.2) for image_path in image_paths]

def computeH(img1_pts, img2_pts):
    A = []
    for (x, y), (u, v) in zip(img1_pts, img2_pts):
        A.append([-x, -y, -1, 0, 0, 0, u*x, u*y, u])
        A.append([0, 0, 0, -x, -y, -1, v*x, v*y, v])
    A = np.array(A)

    # Print the system A·h = 0 as written-out equations
    names = ["h11", "h12", "h13", "h21", "h22", "h23", "h31", "h32", "h33"]
    print("System of equations (A·h = 0):")
    for row in A:
        terms = [f"{c:+.2f}*{n}" for c, n in zip(row, names) if c != 0]
        print(" ".join(terms), "= 0")
    print()

    _, _, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)
    return H / H[2, 2]

def select_points(img1, img2):
    points_img1 = []
    points_img2 = []

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points_img1.append((x, y))
            cv2.circle(img1_copy, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow('Image 1', img1_copy)

    def click_event_img2(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points_img2.append((x, y))
            cv2.circle(img2_copy, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow('Image 2', img2_copy)

    img1_copy = img1.copy()
    img2_copy = img2.copy()

    cv2.imshow('Image 1', img1_copy)
    cv2.setMouseCallback('Image 1', click_event)

    while len(points_img1) < 6:
        cv2.waitKey(1)

    cv2.imwrite(str(RESULTS_DIR / 'selected_points_img1.jpg'), img1_copy)  # Save the image with selected points
    cv2.destroyWindow('Image 1')

    cv2.imshow('Image 2', img2_copy)
    cv2.setMouseCallback('Image 2', click_event_img2)

    while len(points_img2) < 6:
        cv2.waitKey(1)

    cv2.imwrite(str(RESULTS_DIR / 'selected_points_img2.jpg'), img2_copy)  # Save the image with selected points
    cv2.destroyWindow('Image 2')

    return np.array(points_img1), np.array(points_img2)

def main():
    
    images = read_images(IMAGES_DIR)
    
    img1_pts, img2_pts = select_points(images[0], images[1])
    H = computeH(img1_pts, img2_pts)
    print("Computed Homography Matrix:")
    print(H)
    H_cv, _ = cv2.findHomography(img1_pts.astype(np.float32), img2_pts.astype(np.float32))
    print("OpenCV H:\n", H_cv)
    
if __name__ == "__main__":
    main()
