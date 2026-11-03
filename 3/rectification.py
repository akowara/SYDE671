import cv2
import numpy as np
import partA_2 as a2
import partA_3 as a3

img = cv2.imread(str(a2.BASE_DIR / 'rectification' / 'book.jpg'))
img = cv2.resize(img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)

img1_pts, _ = a2.select_points(img, img, num_points=4)
img1_pts = np.asarray(img1_pts, dtype=float)

tl, tr, br, bl = img1_pts
w = int((np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2)
h = int(w / 0.77)

img2_pts = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=float)

H = a2.computeH(img1_pts, img2_pts, show_equations=False)
print("Computed Homography Matrix:")
print(H)

rectified = a3.warpImageBilinear(img, H)
rectified = rectified[:h, :w]
cv2.imwrite(str(a2.RESULTS_DIR / 'rectification.jpg'), rectified)

marked = img.copy()
cv2.polylines(marked, [img1_pts.astype(np.int32)], isClosed=True, color=(0, 0, 255), thickness=2)
for x, y in img1_pts.astype(int):
    cv2.circle(marked, (x, y), 6, (0, 0, 255), -1)
cv2.imwrite(str(a2.RESULTS_DIR / 'rectification_input.jpg'), marked)

img = cv2.imread(str(a2.BASE_DIR / 'rectification' / 'sign.jpeg'))
img = cv2.resize(img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)

img1_pts, _ = a2.select_points(img, img, num_points=4)
img1_pts = np.asarray(img1_pts, dtype=float)

tl, tr, br, bl = img1_pts
w = int((np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2)
h = h = int((np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2)

img2_pts = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=float)

H = a2.computeH(img1_pts, img2_pts, show_equations=False)
print("Computed Homography Matrix:")
print(H)

rectified = a3.warpImageBilinear(img, H)
rectified = rectified[:h, :w]
cv2.imwrite(str(a2.RESULTS_DIR / 'rectification_2.jpg'), rectified)

marked = img.copy()
cv2.polylines(marked, [img1_pts.astype(np.int32)], isClosed=True, color=(0, 0, 255), thickness=2)
for x, y in img1_pts.astype(int):
    cv2.circle(marked, (x, y), 6, (0, 0, 255), -1)
cv2.imwrite(str(a2.RESULTS_DIR / 'rectification_input_2.jpg'), marked)