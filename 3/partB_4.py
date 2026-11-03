import partA_2 as a2
import partA_3 as a3
import partA_4 as a4
import partB_1 as b1
import partB_2 as b2
import partB_3 as b3
import cv2
import numpy as np
import matplotlib.pyplot as plt
import harris as harris

def project(H, pts):
    p = np.hstack([pts, np.ones((len(pts), 1))]) @ H.T
    return p[:, :2] / p[:, 2:3]

def ransac(pts1, pts2, n_iters=2000, eps=2.0, seed=0):
    rng = np.random.default_rng(seed)
    best = np.zeros(len(pts1), dtype=bool)
    for _ in range(n_iters):
        s = rng.choice(len(pts1), 4, replace=False)
        try:
            H = a2.computeH(pts1[s], pts2[s], show_equations=False)
        except np.linalg.LinAlgError:
            continue
        err = np.linalg.norm(project(H, pts1) - pts2, axis=1)
        inliers = err < eps
        if inliers.sum() > best.sum():
            best = inliers

    H = a2.computeH(pts1[best], pts2[best], show_equations=False)
    return H, best

def to_xy(cords, idx):
    return cords[:, idx][::-1].T.astype(float)

def save_titled(path, img, title):
    # Add a white banner with the title above the image, then save it
    scale = max(img.shape[1] / 1000, 0.6)
    thickness = max(int(2 * scale), 1)
    (tw, th), base = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    pad = int(th * 0.8)
    banner = np.full((th + base + 2 * pad, img.shape[1], 3), 255, dtype=np.uint8)
    x = max((img.shape[1] - tw) // 2, 0)
    cv2.putText(banner, title, (x, pad + th), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness, cv2.LINE_AA)
    cv2.imwrite(str(path), np.vstack([banner, img]))

def show_comparison(manual, auto, name):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, im, title in zip(axes, [manual, auto], ["Manual", "Automatic (RANSAC)"]):
        ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle(f"Manual vs Automatic Mosaic (images {name.replace('_', ' & ')}), Bilinear + Laplacian Blend")
    plt.tight_layout()
    plt.savefig(str(a2.RESULTS_DIR / f'comparison_{name}.jpeg'))
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
        matches = b3.match_features(descriptors[k], descriptors[k + 1])
        pts1 = to_xy(anms_corners[k], matches[:, 0])
        pts2 = to_xy(anms_corners[k + 1], matches[:, 1])

        # Same as partA_4: warp image k onto image k+1 (image k+1 stays fixed)
        H, inliers = ransac(pts1, pts2)
        Hs = [H, np.eye(3)]
        print(f"pair {k}: {inliers.sum()} / {len(matches)} inliers")

        b3.show_matches(images[k], images[k + 1], anms_corners[k], anms_corners[k + 1], matches[inliers], f"{k}_{k+1}_ransac")

        warped_nn_auto, alphas_nn = a4.warp_stack(images[k:k+2], Hs , a3.warpImageNearestNeighbor)
        warped_bl_auto, alphas_bl = a4.warp_stack(images[k:k+2], Hs , a3.warpImageBilinear)
            
        out_nn_f = a4.feather_blend(warped_nn_auto, alphas_nn)
        out_bl_f = a4.feather_blend(warped_bl_auto, alphas_bl)
            
        pair = f"images {k} & {k+1}"
        save_titled(f'results/nn_f_mosaic_auto{k}_{k+1}.jpg', out_nn_f, f"Auto mosaic ({pair}): Nearest Neighbor + Feather Blend")
        save_titled(f'results/bl_f_mosaic_auto{k}_{k+1}.jpg', out_bl_f, f"Auto mosaic ({pair}): Bilinear + Feather Blend")

        out_nn_l = a4.laplacian2_blend(warped_nn_auto, alphas_nn)
        out_bl_l = a4.laplacian2_blend(warped_bl_auto, alphas_bl)

        save_titled(f'results/nn_l_mosaic_auto{k}_{k+1}.jpg', out_nn_l, f"Auto mosaic ({pair}): Nearest Neighbor + Laplacian Blend")
        save_titled(f'results/bl_l_mosaic_auto{k}_{k+1}.jpg', out_bl_l, f"Auto mosaic ({pair}): Bilinear + Laplacian Blend")

        manual_path = a2.RESULTS_DIR / f'bl_l_mosaic_{k}.jpg'
        manual = cv2.imread(str(manual_path))
        if manual is None:
            print(f"Missing {manual_path}, run partA_4.py first to compare")
        else:
            show_comparison(manual, out_bl_l, f"{k}_{k+1}")

if __name__ == "__main__":
    main()