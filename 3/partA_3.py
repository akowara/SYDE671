import cv2
import numpy as np
import partA_2 as a2

def warpImageNearestNeighbor(img, H):
    h, w = img.shape[:2]
    H_inv = np.linalg.inv(H)
    
    ys, xs = np.indices((h, w))
    xs, ys = xs.ravel(), ys.ravel()
    pts = np.stack([xs, ys, np.ones_like(xs)], axis=0)
    
    src = H_inv @ pts
    with np.errstate(divide='ignore', invalid='ignore'):
        sx = src[0] / src[2]
        sy = src[1] / src[2]
        
    sx = np.round(sx)
    sy = np.round(sy)
    valid = (sx >= 0) & (sx < w) & (sy >= 0) & (sy < h)
    out = np.zeros_like(img)
    out[ys[valid], xs[valid]] = img[sy[valid].astype(int), sx[valid].astype(int)]
    return out

def warpImageBilinear(img, H):
    h, w = img.shape[:2]
    H_inv = np.linalg.inv(H)
    
    ys, xs = np.indices((h, w))
    xs, ys = xs.ravel(), ys.ravel()
    pts = np.stack([xs, ys, np.ones_like(xs)])
    
    src = H_inv @ pts
    with np.errstate(divide='ignore', invalid='ignore'):
        sx = src[0] / src[2]
        sy = src[1] / src[2]
        
    valid = (sx >= 0) & (sx <= w - 1) & (sy >= 0) & (sy <= h - 1)
    xs, ys, sx, sy = xs[valid], ys[valid], sx[valid], sy[valid]

    x0 = np.floor(sx).astype(int)
    y0 = np.floor(sy).astype(int)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    
    a = sx - x0
    b = sy - y0
    if img.ndim == 3:
        a, b = a[:, None], b[:, None]
        
    imgf = img.astype(np.float32)
    I00 = imgf[y0, x0]
    I01 = imgf[y1, x0]
    I10 = imgf[y0, x1]
    I11 = imgf[y1, x1]
    
    val = (1 - a) * (1 - b) * I00 + (1 - a) * b * I01 + a * (1 - b) * I10 \
        + a * b * I11
        
    out = np.zeros((h, w) + img.shape[2:], dtype=img.dtype)
    out[ys, xs] = np.clip(np.round(val), 0, 255).astype(img.dtype)
    return out

def main():
    
    images = a2.read_images(a2.IMAGES_DIR)
    
    img1_pts, img2_pts = a2.select_points(images[0], images[1])
    H = a2.computeH(img1_pts, img2_pts)
    print("Computed Homography Matrix:")
    print(H)
    H_cv, _ = cv2.findHomography(img1_pts.astype(np.float32), img2_pts.astype(np.float32))
    print("OpenCV H:\n", H_cv)
    
    warped_img = warpImageNearestNeighbor(images[0], H)
    cv2.imwrite(str(a2.RESULTS_DIR / 'warped_nn_image.jpg'), warped_img)
    ref = cv2.warpPerspective(images[0], H, (images[0].shape[1], images[0].shape[0]), flags=cv2.INTER_NEAREST)
    cv2.imwrite(str(a2.RESULTS_DIR / 'reference_nn_image.jpg'), ref)
    print(np.mean(ref != warped_img))
    
    warped_img_bilinear = warpImageBilinear(images[0], H)
    cv2.imwrite(str(a2.RESULTS_DIR / 'warped_bilinear_image.jpg'), warped_img_bilinear)
    ref_bilinear = cv2.warpPerspective(images[0], H, (images[0].shape[1], images[0].shape[0]), flags=cv2.INTER_LINEAR)
    cv2.imwrite(str(a2.RESULTS_DIR / 'reference_bilinear_image.jpg'), ref_bilinear)
    print(np.mean(ref_bilinear != warped_img_bilinear))
    
if __name__ == "__main__":
    main()