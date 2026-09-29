import os
import glob
import cv2
import numpy as np

def read_image(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH)
    img, _ = _to_float(img)
    return img

def crop_white_border(img, threshold=0.9):
    img = img[~np.all((img > threshold) | (img < 1 - threshold), axis=1)]
    img = img[:, ~np.all((img > threshold) | (img < 1 - threshold), axis=0)]
    return img

def crop_black_border(img, threshold=0.04):
    img = img[~np.all((img < threshold) | (img > 1 - threshold), axis=1)]
    img = img[:, ~np.all((img < threshold) | (img > 1 - threshold), axis=0)]
    return img

def split_image(img):
    h, w = img.shape[:2]
    third = h // 3
    remainder = h % 3
    if remainder == 1:
        img = img[:-1, :]
    elif remainder == 2:
        img = img[:-2, :]
    top_third = img[:third, :]
    middle_third = img[third:2*third, :]
    bottom_third = img[2*third:, :]
    return top_third, middle_third, bottom_third

def crop_images(img1, img2, img3):
    imgs = [img1, img2, img3]
    return [crop_image(img) for img in imgs]

def crop_image(img, crop_ratio=0.10):
    height, width = img.shape[:2]  
    crop_h = int(height * crop_ratio)
    crop_w = int(width * crop_ratio)
    img_cropped = img[crop_h:height-crop_h, crop_w:width-crop_w]
    return img_cropped

def merge_images(img1, img2, img3):
    return np.dstack((img1, img2, img3))

def align_channels(img1, img2, img3, algorithm='ncc', shift_size=15, initial_shifts=None, use_edges=False):
    if initial_shifts is None:
        initial_shifts = [(0, 0), (0, 0)]

    imgs = [img2, img3]
    ref = edge_map(img1) if use_edges else img1
    search_imgs = [edge_map(im) for im in imgs] if use_edges else imgs

    cropped_ref = crop_image(ref, crop_ratio=0.10)
    shifts = np.arange(-int(shift_size), int(shift_size) + 1)
    final_imgs = [img1]
    best_shifts = []

    for i, (img, search_img) in enumerate(zip(imgs, search_imgs)):
        best_val = -np.inf
        best_shift = None
        for x in shifts:
            for y in shifts:
                dx = x + initial_shifts[i][0]
                dy = y + initial_shifts[i][1]
                shifted = np.roll(search_img, shift=(dy, dx), axis=(0, 1))

                if algorithm == 'ncc':
                    current_val = ncc(cropped_ref, crop_image(shifted))
                elif algorithm == 'l2':
                    current_val = L2(cropped_ref, crop_image(shifted))

                if current_val > best_val:
                    best_val = current_val
                    best_shift = (dx, dy)

        best_shifts.append(best_shift)
        final_imgs.append(np.roll(img, shift=best_shift[::-1], axis=(0, 1)))

    return final_imgs[0], final_imgs[1], final_imgs[2], best_shifts

def ncc(img1, img2):
    a = img1.astype(np.float64)
    b = img2.astype(np.float64)
    a = (a - a.mean()) / (a.std() + 1e-8)
    b = (b - b.mean()) / (b.std() + 1e-8)
    return np.mean(a * b)

def L2(img1, img2):
    a = img1.astype(np.float64)
    b = img2.astype(np.float64)
    return -np.sum((a - b) ** 2)

def process_image_single(file_path, algorithm="ncc"):
    img = read_image(file_path)
    img = crop_white_border(img)
    img = crop_black_border(img)
    b, g, r = split_image(img)
    b_aligned, g_aligned, r_aligned, _ = align_channels(b, g, r, algorithm=algorithm)
    final_img = merge_images(b_aligned, g_aligned, r_aligned)
    return final_img

def downsample_images(img1, img2, img3, algorithm="ncc", use_edges = False):
    height, width = img1.shape[:2]
    if height < 200 or width < 200:
        img1_aligned, img2_aligned, img3_aligned, best_shifts = align_channels(img1, img2, img3, algorithm=algorithm, shift_size=15, use_edges=use_edges)
        return img1_aligned, img2_aligned, img3_aligned, best_shifts
    else:
        img1_downsampled = cv2.pyrDown(img1)
        img2_downsampled = cv2.pyrDown(img2)
        img3_downsampled = cv2.pyrDown(img3)
        img1_aligned, img2_aligned, img3_aligned, best_shifts= downsample_images(img1_downsampled, img2_downsampled, img3_downsampled, algorithm=algorithm, use_edges=use_edges)
        best_shifts[0] = (best_shifts[0][0] * 2, best_shifts[0][1] * 2)
        best_shifts[1] = (best_shifts[1][0] * 2, best_shifts[1][1] * 2)
        img1_aligned, img2_aligned, img3_aligned, best_shifts = align_channels(img1, img2, img3, algorithm=algorithm, shift_size=2, initial_shifts=best_shifts, use_edges=use_edges)
        return img1_aligned, img2_aligned, img3_aligned, best_shifts

def crop_from_shifts(img, shifts, extra_margin = 0):
    xs = [0] + [shift[0] for shift in shifts]
    ys = [0] + [shift[1] for shift in shifts]

    top = max(ys) +extra_margin
    bottom = -min(ys) +extra_margin
    left = max(xs) +extra_margin
    right = -min(xs) +extra_margin

    h, w = img.shape[:2]
    return img[top:h-bottom, left:w-right]

def process_image_pyramid(file_path, algorithm="ncc"):
    img = read_image(file_path)
    img = crop_white_border(img, threshold=0.78)
    img = crop_black_border(img, threshold=0.22)
    b, g, r = split_image(img)

    b_aligned, g_aligned, r_aligned, best_shifts = downsample_images(b, g, r, algorithm=algorithm)
    final_img = merge_images(b_aligned, g_aligned, r_aligned)
    return final_img, best_shifts

def _to_float(img):
    if np.issubdtype(img.dtype, np.integer):
        return img.astype(np.float64) / np.iinfo(img.dtype).max, img.dtype
    return img.astype(np.float64), img.dtype

def _from_float(f, dtype):
    f = np.clip(f, 0, 1)
    if np.issubdtype(dtype, np.integer):
        return np.round(f * np.iinfo(dtype).max).astype(dtype)
    return f

def auto_contrast(img, low_pct=0, high_pct=100):
    f, dtype = _to_float(img)
    lo = np.percentile(f, low_pct)
    hi = np.percentile(f, high_pct)
    return _from_float((f - lo) / (hi - lo + 1e-8), dtype)

def gamma_correct(img, gamma=0.8):
    f, dtype = _to_float(img)
    return _from_float(f ** gamma, dtype)

def clahe_contrast(img, clip_limit=2.0, tile_size=8):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    y = clahe.apply(y)
    return cv2.cvtColor(cv2.merge((y, cr, cb)), cv2.COLOR_YCrCb2BGR)

def edge_map(img):
    img = img.astype(np.float64)
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    return np.sqrt(gx**2 + gy**2)

def process_image_gradient(file_path, algorithm="ncc"):
    img = read_image(file_path)
    img = crop_white_border(img, threshold=0.78)
    img = crop_black_border(img, threshold=0.22)
    b, g, r = split_image(img)
    b_aligned, g_aligned, r_aligned, best_shifts = downsample_images(b, g, r, algorithm=algorithm, use_edges=True)
    final_img = merge_images(b_aligned, g_aligned, r_aligned)
    
    return final_img, best_shifts

def _border_depth(profile, band, run, window, tol=0.15, dark=0.1, bright=0.95):
    good_run = 0
    for i in range(band):
        inner = np.median(profile[i + run:i + run + window], axis=0)
        line = profile[i]
        bad = np.any(line < dark) or np.any(line > bright) or np.any(np.abs(line - inner) > tol)
        good_run = 0 if bad else good_run + 1
        if good_run == run:
            return i - run + 1
    return band

def auto_crop(img, band_ratio=0.12, margin_ratio=0.005):
    f = img.astype(np.float64)
    h, w = f.shape[:2]
    rows, cols = f.mean(axis=1), f.mean(axis=0)
    bh, bw = int(h * band_ratio), int(w * band_ratio)
    rh, rw = max(3, h // 100), max(3, w // 100)
    wh, ww = max(10, h // 30), max(10, w // 30)
    depths = [_border_depth(rows, bh, rh, wh), _border_depth(rows[::-1], bh, rh, wh),
              _border_depth(cols, bw, rw, ww), _border_depth(cols[::-1], bw, rw, ww)]
    mh, mw = max(1, int(h * margin_ratio)), max(1, int(w * margin_ratio))
    top, bottom, left, right = [d + m if d else 0 for d, m in zip(depths, (mh, mh, mw, mw))]
    return img[top:h - bottom, left:w - right]

def bells_and_whistles(image, best_shifts):
    # image = crop_from_shifts(image, best_shifts)
    image = auto_crop(image)
    image = auto_contrast(image, low_pct=5, high_pct=95)
    # image = _from_float(image, np.dtype(np.uint8))
    # image = clahe_contrast(image)
    return image

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'media/data')
    results_dir = os.path.join(script_dir, 'media/results')
    os.makedirs(results_dir, exist_ok=True)
    algorithm = "ncc"
    for file_path in sorted(glob.glob(os.path.join(data_dir, '*.jpg'))):
        # merged_image = process_image_single(file_path, algorithm=algorithm)
        # merged_image, best_shifts = process_image_pyramid(file_path, algorithm=algorithm)
        merged_image, best_shifts = process_image_gradient(file_path, algorithm=algorithm)
        final_image = bells_and_whistles(merged_image, best_shifts)
        output_path = os.path.join(results_dir, os.path.basename(file_path))
        cv2.imwrite(output_path, _from_float(final_image, np.dtype(np.uint8)))
        print(f'Saved {output_path}')

    for file_path in sorted(glob.glob(os.path.join(data_dir, '*.tif'))):
            # merged_image = process_image_single(file_path, algorithm=algorithm)
            # merged_image, best_shifts = process_image_pyramid(file_path, algorithm=algorithm)
            merged_image, best_shifts = process_image_gradient(file_path, algorithm=algorithm)
            final_image = bells_and_whistles(merged_image, best_shifts)
            output_path = os.path.join(results_dir, os.path.basename(file_path))
            cv2.imwrite(output_path, _from_float(final_image, np.dtype(np.uint16)))
            print(f'Saved {output_path}')


if __name__ == "__main__":
    main()