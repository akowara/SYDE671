import time
import cv2
import numpy as np
import partA_2 as a2

def warpImageNearestNeighbor(img, H, out_shape=None):
    h_in, w_in = img.shape[:2]
    h_out, w_out = out_shape if out_shape is not None else (h_in, w_in)
    H_inv = np.linalg.inv(H)

    ys, xs = np.indices((h_out, w_out))
    xs, ys = xs.ravel(), ys.ravel()
    pts = np.stack([xs, ys, np.ones_like(xs)])
    src = H_inv @ pts
    with np.errstate(divide='ignore', invalid='ignore'):
        sx = np.round(src[0] / src[2])
        sy = np.round(src[1] / src[2])

    valid = (sx >= 0) & (sx < w_in) & (sy >= 0) & (sy < h_in)
    out = np.zeros((h_out, w_out) + img.shape[2:], dtype=img.dtype)
    out[ys[valid], xs[valid]] = img[sy[valid].astype(int), sx[valid].astype(int)]
    return out


def warpImageBilinear(img, H, out_shape=None):
    h_in, w_in = img.shape[:2]
    h_out, w_out = out_shape if out_shape is not None else (h_in, w_in)
    H_inv = np.linalg.inv(H)

    ys, xs = np.indices((h_out, w_out))
    xs, ys = xs.ravel(), ys.ravel()
    pts = np.stack([xs, ys, np.ones_like(xs)])
    src = H_inv @ pts
    with np.errstate(divide='ignore', invalid='ignore'):
        sx = src[0] / src[2]
        sy = src[1] / src[2]

    valid = (sx >= 0) & (sx <= w_in - 1) & (sy >= 0) & (sy <= h_in - 1)
    xs, ys, sx, sy = xs[valid], ys[valid], sx[valid], sy[valid]

    x0 = np.floor(sx).astype(int)
    y0 = np.floor(sy).astype(int)
    x1 = np.minimum(x0 + 1, w_in - 1)
    y1 = np.minimum(y0 + 1, h_in - 1)
    a = sx - x0
    b = sy - y0
    if img.ndim == 3:
        a, b = a[:, None], b[:, None]

    imgf = img.astype(np.float32)
    val = (1 - a) * (1 - b) * imgf[y0, x0] + a * (1 - b) * imgf[y0, x1] \
        + (1 - a) * b * imgf[y1, x0] + a * b * imgf[y1, x1]

    out = np.zeros((h_out, w_out) + img.shape[2:], dtype=img.dtype)
    if np.issubdtype(img.dtype, np.integer):
        out[ys, xs] = np.clip(np.round(val), 0, 255).astype(img.dtype)
    else:
        out[ys, xs] = val
    return out


def main():
    
    images = a2.read_images(a2.IMAGES_DIR)
    print(images[0].shape, images[1].shape, images[2].shape, images[3].shape)
    for i in range(0, len(images) -1, 2):
        img1_pts, img2_pts = a2.select_points(images[i], images[i + 1], i)
        H = a2.computeH(img1_pts, img2_pts, show_equations=False)
        print("Computed Homography Matrix:")
        print(H)
        
        t0 = time.perf_counter()
        warped_img = warpImageNearestNeighbor(images[i], H)
        print(f"warpImageNearestNeighbor time: {time.perf_counter() - t0:.4f} s")
        cv2.imwrite(str(a2.RESULTS_DIR / f'warped_nn_image_{i}.jpg'), warped_img)
        t0 = time.perf_counter()
        ref = cv2.warpPerspective(images[i], H, (images[i].shape[1], images[i].shape[0]), flags=cv2.INTER_NEAREST)
        print(f"cv2.warpPerspective (nearest) time: {time.perf_counter() - t0:.4f} s")
        cv2.imwrite(str(a2.RESULTS_DIR / f'reference_nn_image_{i}.jpg'), ref)
        print(np.mean(ref != warped_img))
        
        t0 = time.perf_counter()
        warped_img_bilinear = warpImageBilinear(images[i], H)
        print(f"warpImageBilinear time: {time.perf_counter() - t0:.4f} s")
        cv2.imwrite(str(a2.RESULTS_DIR / f'warped_bilinear_image_{i}.jpg'), warped_img_bilinear)
        t0 = time.perf_counter()
        ref_bilinear = cv2.warpPerspective(images[i], H, (images[i].shape[1], images[i].shape[0]), flags=cv2.INTER_LINEAR)
        print(f"cv2.warpPerspective (bilinear) time: {time.perf_counter() - t0:.4f} s")
        cv2.imwrite(str(a2.RESULTS_DIR / f'reference_bilinear_image_{i}.jpg'), ref_bilinear)
        print(np.mean(ref_bilinear != warped_img_bilinear))
    
if __name__ == "__main__":
    main()