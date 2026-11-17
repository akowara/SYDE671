import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
import scipy.io as sio
import matplotlib.gridspec as gridspec
from epipolar_utils import *

np.set_printoptions(precision=4, suppress=True)

'''
FACTORIZATION_METHOD The Tomasi and Kanade Factorization Method to determine
the 3D structure of the scene and the motion of the cameras.
Arguments:
    points_im1 - N points in the first image that match with points_im2
    points_im2 - N points in the second image that match with points_im1

    Both points_im1 and points_im2 are from the get_data_from_txt_file() method
Returns:
    structure - the structure matrix
    motion - the motion matrix
'''
def factorization_method(points_im1, points_im2):
    centroid1 = np.mean(points_im1[:, :2], axis = 0)
    centroid2 = np.mean(points_im2[:, :2], axis = 0)
    pts1 = points_im1[:, :2] - centroid1
    pts2 = points_im2[:, :2] - centroid2
    D = np.vstack((pts1.T, pts2.T))
    U, s, Vh = np.linalg.svd(D, full_matrices = False)
    U3 = U[:, :3]
    
    S3 = np.diag(s[:3])
    Vh3 = Vh[:3]

    motion = U3 @np.sqrt(S3)
    structure = np.sqrt(S3) @ Vh3


    print(s)

    return structure, motion
    


if __name__ == '__main__':
    for im_set in ['data/set1', 'data/set1_subset']:
        # Read in the data
        im1 = imread(im_set+'/image1.jpg')
        im2 = imread(im_set+'/image2.jpg')
        points_im1 = get_data_from_txt_file(im_set + '/pt_2D_1.txt')
        points_im2 = get_data_from_txt_file(im_set + '/pt_2D_2.txt')
        points_3d = get_data_from_txt_file(im_set + '/pt_3D.txt')
        assert (points_im1.shape == points_im2.shape)

        # Run the Factorization Method
        structure, motion = factorization_method(points_im1, points_im2)

        # Plot the structure
        fig = plt.figure()
        ax = fig.add_subplot(121, projection = '3d')
        scatter_3D_axis_equal(structure[0,:], structure[1,:], structure[2,:], ax)
        ax.set_title('Factorization Method')
        ax = fig.add_subplot(122, projection = '3d')
        scatter_3D_axis_equal(points_3d[:,0], points_3d[:,1], points_3d[:,2], ax)
        ax.set_title('Ground Truth')

        plt.show()
