import partA_2 as a2
import partA_3 as a3
import cv2
import numpy as np

def make_alpha(h, w):
    m = np.zeros((h+2, w+2), dtype=np.uint8)
    m[1:-1, 1:-1] = 1
    d = cv2.distanceTransform(m, cv2.DIST_L2, 5)[1:-1, 1:-1]
    return (d/d.max()).astype(np.float32)

def mosaic_canvas(images, Hs):
    corners = []
    for img, H in zip(images, Hs):
        h, w = img.shape[:2]
        c = H @ np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]]).T
        corners.append(c[:2] / c[2])
    c = np.hstack(corners)
    x_min, y_min = np.floor(c.min(axis=1)).astype(int)
    x_max, y_max = np.ceil(c.max(axis=1)).astype(int)
    T = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype = float)
    return T, (y_max - y_min, x_max - x_min)

def warp_stack(images, Hs, warpFunction):
    T, shape = mosaic_canvas(images, Hs)
    warped, alphas = [], []
    for img, H in zip(images, Hs):
        h, w = img.shape[:2]
        warped.append(warpFunction(img.astype(np.float32), T @ H, shape))
        alphas.append(warpFunction(make_alpha(h, w), T @ H, shape))
    return warped, alphas

def feather_blend(warped, alphas):
    num = sum(img * a[..., None] for img, a in zip(warped, alphas))
    den = sum(alphas)[..., None]
    out = num / np.maximum(den, 1e-8)
    return np.clip(out, 0, 255).astype(np.uint8)

def laplacian2_blend(warped, alphas, sigma=5):
    lows = [cv2.GaussianBlur(img, (0, 0), sigma) for img in warped]
    highs = [img - low for img, low in zip(warped, lows)]

    num = sum(low * a[..., None] for low, a in zip(lows, alphas))
    den = np.maximum(sum(alphas), 1e-8)[..., None]
    low_blend = num / den

    winner = np.argmax(np.stack(alphas), axis=0)
    high_blend = np.zeros_like(warped[0])
    for i, h in enumerate(highs):
        high_blend[winner == i] = h[winner == i]

    return np.clip(low_blend + high_blend, 0, 255).astype(np.uint8)

def main():
        
    images = a2.read_images(a2.IMAGES_DIR)

    for i in range(0, len(images) -1, 2):
        img1_pts, img2_pts = a2.select_points(images[i], images[i + 1], i)
        H12 = a2.computeH(img1_pts, img2_pts, show_equations=False)
        Hs = [H12, np.eye(3)]
        
        warped_nn, alphas_nn = warp_stack(images[i:i + 2], Hs, a3.warpImageNearestNeighbor)
        warped_bl, alphas_bl = warp_stack(images[i:i + 2], Hs, a3.warpImageBilinear)
        
        out_nn_f = feather_blend(warped_nn, alphas_nn)
        out_bl_f = feather_blend(warped_bl, alphas_bl)
        
        cv2.imwrite(f'results/nn_f_mosaic_{i}.jpg', out_nn_f)
        cv2.imwrite(f'results/bl_f_mosaic_{i}.jpg', out_bl_f)
        
        out_nn_l = laplacian2_blend(warped_nn, alphas_nn)
        out_bl_l = laplacian2_blend(warped_bl, alphas_bl)
            
        cv2.imwrite(f'results/nn_l_mosaic_{i}.jpg', out_nn_l)
        cv2.imwrite(f'results/bl_l_mosaic_{i}.jpg', out_bl_l)
        
        # cv2.imwrite('mosaic_laplacian.jpg', laplacian2_blend(warped, alphas))
        
        # stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
        # status, pano = stitcher.stitch([images[i], images[i + 1]])
        # if status == cv2.Stitcher_OK:
        #     cv2.imwrite(f'results/reference_stitcher_{i}.jpg', pano)
        # else:
        #     print("Stitcher failed, status", status) 
    
if __name__ == "__main__":
    main()
    
    
