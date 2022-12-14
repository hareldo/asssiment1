"""Projective Homography and Panorama Solution."""
import numpy as np

from typing import Tuple
from random import sample
from collections import namedtuple

from numpy.linalg import svd
from scipy.interpolate import griddata

PadStruct = namedtuple('PadStruct',
                       ['pad_up', 'pad_down', 'pad_right', 'pad_left'])


class Solution:
    """Implement Projective Homography and Panorama Solution."""

    def __init__(self):
        pass

    @staticmethod
    def compute_homography_naive(match_p_src: np.ndarray,
                                 match_p_dst: np.ndarray) -> np.ndarray:
        """Compute a Homography in the Naive approach, using SVD decomposition.

        Args:
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.

        Returns:
            Homography from source to destination, 3x3 numpy array.
        """
        N = match_p_src.shape[1]
        A = np.zeros([2 * N, 9])

        u_v_one = np.ones([N, 3])
        u_v_one[:, :2] = match_p_src.T
        A[::2, :3] = -1 * u_v_one
        A[1::2, 3:6] = -1 * u_v_one
        A[::2, 6:] = (u_v_one.T * match_p_dst[0]).T
        A[1::2, 6:] = (u_v_one.T * match_p_dst[1]).T
        U, S, V = np.linalg.svd(A)
        H = V[-1, :].reshape((3, 3))
        return H / H[-1, -1]

    @staticmethod
    def compute_forward_homography_slow(
            homography: np.ndarray,
            src_image: np.ndarray,
            dst_image_shape: tuple = (1088, 1452, 3)) -> np.ndarray:
        """Compute a Forward-Homography in the Naive approach, using loops.

        Iterate over the rows and columns of the source image, and compute
        the corresponding point in the destination image using the
        projective homography. Place each pixel value from the source image
        to its corresponding location in the destination image.
        Don't forget to round the pixel locations computed using the
        homography.

        Args:
            homography: 3x3 Projective Homography matrix.
            src_image: HxWx3 source image.
            dst_image_shape: tuple of length 3 indicating the destination
            image height, width and color dimensions.

        Returns:
            The forward homography of the source image to its destination.
        """
        new_image = np.zeros(dst_image_shape)
        h_src, w_src = src_image.shape[:2]
        for v in range(h_src):
            for u in range(w_src):
                x = np.array([u, v, 1])
                x_tag = homography.dot(x)
                x_tag = (x_tag / x_tag[2]).astype(int)

                u_dst = max(x_tag[0], 0)
                u_dst = min(u_dst, dst_image_shape[1] - 1)
                v_dst = max(x_tag[1], 0)
                v_dst = min(v_dst, dst_image_shape[0] - 1)

                new_image[v_dst, u_dst, :] = src_image[v, u, :] / 255

        return new_image

    @staticmethod
    def compute_forward_homography_fast(
            homography: np.ndarray,
            src_image: np.ndarray,
            dst_image_shape: tuple = (1088, 1452, 3)) -> np.ndarray:
        """Compute a Forward-Homography in a fast approach, WITHOUT loops.

        (1) Create a meshgrid of columns and rows.
        (2) Generate a matrix of size 3x(H*W) which stores the pixel locations
        in homogeneous coordinates.
        (3) Transform the source homogeneous coordinates to the target
        homogeneous coordinates with a simple matrix multiplication and
        apply the normalization you've seen in class.
        (4) Convert the coordinates into integer values and clip them
        according to the destination image size.
        (5) Plant the pixels from the source image to the target image according
        to the coordinates you found.

        Args:
            homography: 3x3 Projective Homography matrix.
            src_image: HxWx3 source image.
            dst_image_shape: tuple of length 3 indicating the destination.
            image height, width and color dimensions.

        Returns:
            The forward homography of the source image to its destination.
        """
        h_src, w_src = src_image.shape[:2]
        xx, yy = np.meshgrid(range(w_src), range(h_src), sparse=False, indexing='xy')
        src_grid = np.vstack([xx.flatten(), yy.flatten(), np.ones(h_src * w_src)])

        src_to_dst_grid = np.dot(homography, src_grid)
        src_to_dst_grid = (src_to_dst_grid / src_to_dst_grid[2, :]).astype('int')
        src_to_dst_grid[0, :] = src_to_dst_grid[0, :].clip(min=0, max=dst_image_shape[1] - 1)
        src_to_dst_grid[1, :] = src_to_dst_grid[1, :].clip(min=0, max=dst_image_shape[0] - 1)
        src_to_dst_grid = src_to_dst_grid.reshape((3, h_src, w_src))

        src_grid = src_grid.reshape((3, h_src, w_src)).astype('int')
        dst_image = np.zeros(dst_image_shape)
        dst_image[src_to_dst_grid[1], src_to_dst_grid[0]] = src_image[src_grid[1], src_grid[0]]

        return dst_image.astype('int')

    @staticmethod
    def test_homography(homography: np.ndarray,
                        match_p_src: np.ndarray,
                        match_p_dst: np.ndarray,
                        max_err: float) -> Tuple[float, float]:
        """Calculate the quality of the projective transformation model.

        Args:
            homography: 3x3 Projective Homography matrix.
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.
            max_err: A scalar that represents the maximum distance (in
            pixels) between the mapped src point to its corresponding dst
            point, in order to be considered as valid inlier.

        Returns:
            A tuple containing the following metrics to quantify the
            homography performance:
            fit_percent: The probability (between 0 and 1) validly mapped src
            points (inliers).
            dist_mse: Mean square error of the distances between validly
            mapped src points, to their corresponding dst points (only for
            inliers). In edge case where the number of inliers is zero,
            return dist_mse = 10 ** 9.
        """
        # add ones to source pts to perform matrix multi for affinite matrix
        pts_shape = match_p_src.shape[1]

        # move math points from source to dst coodicnate
        affinite_result = homography.dot(np.concatenate((match_p_src, np.ones((1, pts_shape))), axis=0))

        # find the true vectors
        u_i = np.divide(affinite_result[0, :], affinite_result[2, :])
        v_i = np.divide(affinite_result[1, :], affinite_result[2, :])

        # source pts in destination coordinate
        pts_src_in_dst_coord = np.array([u_i, v_i])

        # compute distance between two points in destuination image coordonates
        distance = np.apply_along_axis(np.linalg.norm, arr=(match_p_dst - pts_src_in_dst_coord), axis=0)
        inlier_valid_dist = distance[distance <= max_err]

        # fit_percent = len(inlier_valid) / len(distance)
        fit_percent = inlier_valid_dist.shape[0] / distance.shape[0]

        if inlier_valid_dist.shape[0] != 0:
            # compute MSE
            dist_mse = np.mean(np.square(inlier_valid_dist))
        else:
            dist_mse = 10 ** 9

        return fit_percent, dist_mse

    @staticmethod
    def meet_the_model_points(homography: np.ndarray,
                              match_p_src: np.ndarray,
                              match_p_dst: np.ndarray,
                              max_err: float) -> Tuple[np.ndarray, np.ndarray]:
        """Return which matching points meet the homography.

        Loop through the matching points, and return the matching points from
        both images that are inliers for the given homography.

        Args:
            homography: 3x3 Projective Homography matrix.
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.
            max_err: A scalar that represents the maximum distance (in
            pixels) between the mapped src point to its corresponding dst
            point, in order to be considered as valid inlier.
        Returns:
            A tuple containing two numpy nd-arrays, containing the matching
            points which meet the model (the homography). The first entry in
            the tuple is the matching points from the source image. That is a
            nd-array of size 2xD (D=the number of points which meet the model).
            The second entry is the matching points form the destination
            image (shape 2xD; D as above).
        """
        # add ones to source pts to perform matrix multi for affinite matrix
        pts_shape = match_p_src.shape[1]

        # move math points from source to dst coodicnate
        affinite_result = homography.dot(np.concatenate((match_p_src, np.ones((1, pts_shape))), axis=0))

        # find the true vectors
        u_i = np.divide(affinite_result[0, :], affinite_result[2, :])
        v_i = np.divide(affinite_result[1, :], affinite_result[2, :])

        # source pts in destination coordinate
        pts_src_in_dst_coord = np.array([u_i, v_i])

        # compute distance between two points in destuination image coordonates
        distance = np.apply_along_axis(np.linalg.norm, arr=(match_p_dst - pts_src_in_dst_coord), axis=0)

        # get matching points only
        mp_src_meets_model = match_p_src[:, distance <= max_err]
        mp_dst_meets_model = match_p_dst[:, distance <= max_err]

        return mp_src_meets_model, mp_dst_meets_model

    def compute_homography(self,
                           match_p_src: np.ndarray,
                           match_p_dst: np.ndarray,
                           inliers_percent: float,
                           max_err: float) -> np.ndarray:
        """Compute homography coefficients using RANSAC to overcome outliers.

        Args:
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.
            inliers_percent: The expected probability (between 0 and 1) of
            correct match points from the entire list of match points.
            max_err: A scalar that represents the maximum distance (in
            pixels) between the mapped src point to its corresponding dst
            point, in order to be considered as valid inlier.
        Returns:
            homography: Projective transformation matrix from src to dst.
        """
        # use class notations:
        w = inliers_percent
        # t = max_err
        # p = parameter determining the probability of the algorithm to
        # succeed
        p = 0.99
        # the minimal probability of points which meets with the model
        d = 0.5
        # number of points sufficient to compute the model
        n = 4
        # number of RANSAC iterations (+1 to avoid the case where w=1)
        k = int(np.ceil(np.log(1 - p) / np.log(1 - w ** n))) + 1

        # variables declaration
        min_dist_mse = 10 ** 9
        homography = 0
        one_model_found = 0
        pts_shape = match_p_src.shape[1]

        for i in range(k):
            # Array for random sampling
            sample_arr = [True, False]

            # Create a numpy array with random True or False of size 10
            bool_arr = np.random.choice(sample_arr, size=pts_shape)

            # current random choosen points
            curr_match_p_src = match_p_src[:, bool_arr]
            curr_match_p_dst = match_p_dst[:, bool_arr]

            # compute and test homography
            curr_homography = self.compute_homography_naive(curr_match_p_src, curr_match_p_dst)
            fit_percent, dist_mse = self.test_homography(curr_homography, curr_match_p_src[:, :],
                                                         curr_match_p_dst[:, :], max_err)

            if fit_percent > d:
                # get all inliers according to current homography
                mp_src_meets_model, mp_dst_meets_model = self.meet_the_model_points(curr_homography, match_p_src[:, :],
                                                                                    match_p_dst[:, :], max_err)

                # recompute the model using all inliers
                valid_curr_homography = self.compute_homography_naive(mp_src_meets_model, mp_dst_meets_model)

                # test the current homography with all the founded inliers
                fit_percent_all, dist_mse_all = self.test_homography(valid_curr_homography, mp_src_meets_model[:, :],
                                                                     mp_dst_meets_model[:, :],
                                                                     max_err)

                # save the best model homography
                if dist_mse_all < min_dist_mse:
                    min_dist_mse = dist_mse_all
                    homography = curr_homography
                    one_model_found = 1

        if one_model_found == 0:
            print('RANSAC algorithm didnt find any sufficient model with those parameters \n exit...')
            exit()

        return homography

    @staticmethod
    def compute_backward_mapping(
            backward_projective_homography: np.ndarray,
            src_image: np.ndarray,
            dst_image_shape: tuple = (1088, 1452, 3)) -> np.ndarray:
        """Compute backward mapping.

        (1) Create a mesh-grid of columns and rows of the destination image.
        (2) Create a set of homogenous coordinates for the destination image
        using the mesh-grid from (1).
        (3) Compute the corresponding coordinates in the source image using
        the backward projective homography.
        (4) Create the mesh-grid of source image coordinates.
        (5) For each color channel (RGB): Use scipy's interpolation.griddata
        with an appropriate configuration to compute the bi-cubic
        interpolation of the projected coordinates.

        Args:
            backward_projective_homography: 3x3 Projective Homography matrix.
            src_image: HxWx3 source image.
            dst_image_shape: tuple of length 3 indicating the destination shape.

        Returns:
            The source image backward warped to the destination coordinates.
        """
        h_dst, w_dst = dst_image_shape[:2]
        xx_d, yy_d = np.meshgrid(range(w_dst), range(h_dst), sparse=False, indexing='xy')
        dst_grid = np.vstack([xx_d.flatten(), yy_d.flatten(), np.ones(h_dst * w_dst)])

        dst_to_src_grid = np.dot(backward_projective_homography, dst_grid)
        dst_to_src_grid = (dst_to_src_grid / dst_to_src_grid[2, :]).astype('int')
        dst_to_src_grid[0, :] = dst_to_src_grid[0, :].clip(min=0, max=w_dst - 1)
        dst_to_src_grid[1, :] = dst_to_src_grid[1, :].clip(min=0, max=h_dst - 1)
        dst_to_src_grid = dst_to_src_grid.reshape((3, h_dst, w_dst))
        yy_d = dst_to_src_grid[1]
        xx_d = dst_to_src_grid[0]

        h_src, w_src = src_image.shape[:2]
        xx_s, yy_s = np.meshgrid(range(w_src), range(h_src), sparse=False, indexing='xy')
        xx_s = xx_s.flatten()
        yy_s = yy_s.flatten()
        backward_warped = griddata((yy_s, xx_s), src_image[yy_s, xx_s], (yy_d, xx_d), method='cubic')
        return backward_warped

    @staticmethod
    def find_panorama_shape(src_image: np.ndarray,
                            dst_image: np.ndarray,
                            homography: np.ndarray
                            ) -> Tuple[int, int, PadStruct]:
        """Compute the panorama shape and the padding in each axes.

        Args:
            src_image: Source image expected to undergo projective
            transformation.
            dst_image: Destination image to which the source image is being
            mapped to.
            homography: 3x3 Projective Homography matrix.

        For each image we define a struct containing it's corners.
        For the source image we compute the projective transformation of the
        coordinates. If some of the transformed image corners yield negative
        indices - the resulting panorama should be padded with at least
        this absolute amount of pixels.
        The panorama's shape should be:
        dst shape + |the largest negative index in the transformed src index|.

        Returns:
            The panorama shape and a struct holding the padding in each axes (
            row, col).
            panorama_rows_num: The number of rows in the panorama of src to dst.
            panorama_cols_num: The number of columns in the panorama of src to
            dst.
            padStruct = a struct with the padding measures along each axes
            (row,col).
        """
        src_rows_num, src_cols_num, _ = src_image.shape
        dst_rows_num, dst_cols_num, _ = dst_image.shape
        src_edges = {}
        src_edges['upper left corner'] = np.array([1, 1, 1])
        src_edges['upper right corner'] = np.array([src_cols_num, 1, 1])
        src_edges['lower left corner'] = np.array([1, src_rows_num, 1])
        src_edges['lower right corner'] = \
            np.array([src_cols_num, src_rows_num, 1])
        transformed_edges = {}
        for corner_name, corner_location in src_edges.items():
            transformed_edges[corner_name] = homography @ corner_location
            transformed_edges[corner_name] /= transformed_edges[corner_name][-1]
        pad_up = pad_down = pad_right = pad_left = 0
        for corner_name, corner_location in transformed_edges.items():
            if corner_location[1] < 1:
                # pad up
                pad_up = max([pad_up, abs(corner_location[1])])
            if corner_location[0] > dst_cols_num:
                # pad right
                pad_right = max([pad_right,
                                 corner_location[0] - dst_cols_num])
            if corner_location[0] < 1:
                # pad left
                pad_left = max([pad_left, abs(corner_location[0])])
            if corner_location[1] > dst_rows_num:
                # pad down
                pad_down = max([pad_down,
                                corner_location[1] - dst_rows_num])
        panorama_cols_num = int(dst_cols_num + pad_right + pad_left)
        panorama_rows_num = int(dst_rows_num + pad_up + pad_down)
        pad_struct = PadStruct(pad_up=int(pad_up),
                               pad_down=int(pad_down),
                               pad_left=int(pad_left),
                               pad_right=int(pad_right))
        return panorama_rows_num, panorama_cols_num, pad_struct

    @staticmethod
    def add_translation_to_backward_homography(backward_homography: np.ndarray,
                                               pad_left: int,
                                               pad_up: int) -> np.ndarray:
        """Create a new homography which takes translation into account.

        Args:
            backward_homography: 3x3 Projective Homography matrix.
            pad_left: number of pixels that pad the destination image with
            zeros from left.
            pad_up: number of pixels that pad the destination image with
            zeros from the top.

        (1) Build the translation matrix from the pads.
        (2) Compose the backward homography and the translation matrix together.
        (3) Scale the homography as learnt in class.

        Returns:
            A new homography which includes the backward homography and the
            translation.
        """
        translation_mat = np.array([[1, 0, -pad_left], [0, 1, -pad_up], [0, 0, 1]])
        final_homography = np.dot(backward_homography, translation_mat)
        final_homography /= np.linalg.norm(final_homography)
        return final_homography

    def panorama(self,
                 src_image: np.ndarray,
                 dst_image: np.ndarray,
                 match_p_src: np.ndarray,
                 match_p_dst: np.ndarray,
                 inliers_percent: float,
                 max_err: float) -> np.ndarray:
        """Produces a panorama image from two images, and two lists of
        matching points, that deal with outliers using RANSAC.

        (1) Compute the forward homography and the panorama shape.
        (2) Compute the backward homography.
        (3) Add the appropriate translation to the homography so that the
        source image will plant in place.
        (4) Compute the backward warping with the appropriate translation.
        (5) Create the an empty panorama image and plant there the
        destination image.
        (6) place the backward warped image in the indices where the panorama
        image is zero.
        (7) Don't forget to clip the values of the image to [0, 255].


        Args:
            src_image: Source image expected to undergo projective
            transformation.
            dst_image: Destination image to which the source image is being
            mapped to.
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.
            inliers_percent: The expected probability (between 0 and 1) of
            correct match points from the entire list of match points.
            max_err: A scalar that represents the maximum distance (in pixels)
            between the mapped src point to its corresponding dst point,
            in order to be considered as valid inlier.

        Returns:
            A panorama image.

        """
        forward_homography = self.compute_homography(match_p_src, match_p_dst, inliers_percent, max_err)
        backward_homography = self.compute_homography(match_p_dst, match_p_src, inliers_percent, max_err)
        rows, cols, pad_struct = Solution.find_panorama_shape(src_image, dst_image, forward_homography)
        backward_homography = Solution.add_translation_to_backward_homography(backward_homography,
                                                                              pad_struct.pad_left,
                                                                              pad_struct.pad_up)

        panorama = np.zeros(shape=(rows, cols, 3), dtype=np.uint8)
        panorama[pad_struct.pad_up: pad_struct.pad_up + dst_image.shape[0],
        pad_struct.pad_left: pad_struct.pad_left + dst_image.shape[1]] = dst_image

        backward_warped = Solution.compute_backward_mapping(backward_homography, src_image, panorama.shape)
        mask = panorama[: backward_warped.shape[0], :backward_warped.shape[1]] == [0, 0, 0]
        panorama[mask] = backward_warped[mask]
        return panorama
