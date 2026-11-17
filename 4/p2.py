import numpy as np
import matplotlib.pyplot as plt
from p1 import *
from epipolar_utils import *

'''
COMPUTE_EPIPOLE computes the epipole e in homogenous coordinates
given the fundamental matrix
Arguments:
    F - the Fundamental matrix solved for with normalized_eight_point_alg(points1, points2)

Returns:
    epipole - the homogenous coordinates [x y 1] of the epipole in the image
'''
def compute_epipole(F):
    U, S, V = np.linalg.svd(F)
    e = V[-1]
    e = e/e[2]
    return e

'''
COMPUTE_H computes a homography to map an epipole to infinity along the horizontal axis 
Arguments:
    e - the epipole
    im2 - the image
Returns:
    H - homography matrix
'''
def compute_H(e, im):
    h,w = im.shape[:2]
    e = e/e[2]
    T = np.array([[1, 0, -(w/2)],
                 [0, 1, -(h/2)],
                 [0, 0, 1]])
    e1, e2, _ = T @ e
    denom = np.sqrt(e1**2 + e2**2)
    alpha = 1 if e1>=0 else -1
    R = np.array([[alpha*(e1/denom), alpha*(e2/denom), 0],
                 [-1*alpha*(e2/denom), alpha*(e1/denom), 0],
                 [0, 0, 1]])
    def g_func(f):
        return np.array([[1, 0, 0],
                        [0, 1, 0],
                        [-1/f, 0, 1]])
    f = alpha*denom
    G = g_func(f)
    H = np.linalg.inv(T) @ G @ R @ T
    print(H @ e)
    return H
  
'''
COMPUTE_MATCHING_HOMOGRAPHIES determines homographies H1 and H2 such that they
rectify a pair of images
Arguments:
    e2 - the second epipole
    F - the Fundamental matrix
    im2 - the second image
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1
Returns:
    H1 - the homography associated with the first image
    H2 - the homography associated with the second image
'''
def compute_matching_homographies(e2, F, im2, points1, points2):
    H2 = compute_H(e2, im2)
    v = np.array([1, 1, 1])
    x, y, w = e2
    e_cross = np.array([[0, -w, y],
               [w, 0, -x],
               [-y, x, 0]])
    M = e_cross @ F + np.outer(e2, v)

    pts1 = (H2@M@points1.T).T
    pts1 = pts1/pts1[:,2:3]
    pts2 = (H2@points2.T).T
    pts2 = pts2/pts2[:,2:3]

    W = np.column_stack((pts1[:,0], pts1[:, 1], np.ones_like(pts1[:,0])))
    b = pts2[:,0]
    a, res, r, s = np.linalg.lstsq(W, b, rcond=None)

    HA = np.array([[a[0], a[1], a[2]],
                    [0, 1, 0],
                    [0, 0, 1]])

    H1 = HA@H2@M
    return H1, H2
   




if __name__ == '__main__':
    # Read in the data
    im_set = 'data/set1'
    im1 = imread(im_set+'/image1.jpg')
    im2 = imread(im_set+'/image2.jpg')
    points1 = get_data_from_txt_file(im_set+'/pt_2D_1.txt')
    points2 = get_data_from_txt_file(im_set+'/pt_2D_2.txt')
    assert (points1.shape == points2.shape)

    F = normalized_eight_point_alg(points1, points2)
    # F is such that such that (points2)^T * F * points1 = 0, so e1 is e' and e2 is e
    e1 = compute_epipole(F.T)
    e2 = compute_epipole(F)
    print("e1", e1)
    print("e2", e2)

    # Find the homographies needed to rectify the pair of images
    H1, H2 = compute_matching_homographies(e2, F, im2, points1, points2)
    print('')

    # Transforming the images by the homographies
    new_points1 = H1.dot(points1.T)
    new_points2 = H2.dot(points2.T)
    new_points1 /= new_points1[2,:]
    new_points2 /= new_points2[2,:]
    new_points1 = new_points1.T
    new_points2 = new_points2.T
    rectified_im1, offset1 = compute_rectified_image(im1, H1)
    rectified_im2, offset2 = compute_rectified_image(im2, H2)
    new_points1 -= offset1 + (0,)
    new_points2 -= offset2 + (0,)

    # Plotting the image
    F_new = normalized_eight_point_alg(new_points1, new_points2)
    plot_epipolar_lines_on_images(new_points1, new_points2, rectified_im1, rectified_im2, F_new)
    plt.show()
