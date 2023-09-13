import numpy as np


def preprocess_images(imgs_raw, oversaturation_thresh=2 ** 16 - 1):
    """
    The preprocessing of image stacks is composed of the following steps:
    1. Substract a mask [the mask is based on the median]
    2. Filter out images where the mean of their positive values is no more than some thresholds, and the same for their
       negative values
    3. Add a constant to make the images non-negative, then set any oversaturating pixels to the oversaturation
       threshold
    4. Return images and also the kept indices

    Args:
        imgs_raw (ndarray): N x height x width stack of images (typically np.uint16 pixels)
        oversaturation_thresh (int): see step 3 above.
    Returns:
        imgs_out (ndarray): M x height x width stack of preprocessed images where M<=N. pixels are always np.uint16.
        inds_in_out (ndarray): N x 2 ndarray where the first column is just 0 to N-1 and the second column is the
        indices of the images in the new file: if image n survived the filtering then inds_in_out[n,1] is m<n such that
        imgs_out[m] corresponds to the processed imgs_raw[n]. If it didn't survive the filter then inds_in_out[n,1]=nan.
    """
    N_images_orig = imgs_raw.shape[0]

    subtract_mask, pos_neg_thresholds = gen_params_for_preprocess(imgs_raw, percentile=85,
                                                                  oversaturation_thresh=oversaturation_thresh)

    imgs = imgs_raw.astype(np.float32)
    # Step 1
    imgs_med_sub = imgs - subtract_mask
    mean_pos_arr = np.mean(np.maximum(imgs_med_sub, 0), axis=(1, 2))
    mean_neg_arr = np.mean(np.minimum(imgs_med_sub, 0), axis=(1, 2))

    # Step 2
    inds_to_keep_mask = np.logical_and(mean_pos_arr <= pos_neg_thresholds[0], mean_neg_arr >= pos_neg_thresholds[1])
    print(
        f"Kept {np.sum(inds_to_keep_mask)}/{N_images_orig} images in the preprocessing ({np.sum(inds_to_keep_mask) / N_images_orig:.2f})")

    # Step 3
    imgs_med_sub_positive = imgs_med_sub + np.max(subtract_mask)
    imgs_out = np.minimum(imgs_med_sub_positive[inds_to_keep_mask, ...], oversaturation_thresh).astype(np.uint16)

    # Step 4
    inds_in = np.arange(N_images_orig, dtype=int)
    inds_out = inds_in + np.nan
    count = 0
    for i in range(len(inds_out)):
        if inds_to_keep_mask[i]:
            inds_out[i] = count
            count += 1
    inds_in_out = np.vstack([inds_in, inds_out])
    return imgs_out, inds_in_out


def gen_params_for_preprocess(imgs_raw, percentile, oversaturation_thresh):
    """
    Generate parameters subtract_mask, pos_neg_thresholds for function preprocess_images.
    This calculates:
    1. The mask that is to be subtracted from the images (approximately the median mask)
    2. The threshold for the mean positive values and for the mean negative values of the subtracted image

    Args:
        imgs_raw (ndarray): N x height x width stack of images (typically np.uint16 pixels)
        percentile (int): approximate percentage of images to keep in the process of filtering
        oversaturation_thresh (int): see preprocess_images()
    Returns:
        subtract_mask (ndarray): height x width stack that is approximately the median
        pos_neg_thresholds (ndarray): len 2 array with the positive threshold and negative threshold for keeping images
        (see preprocess_images()).

    """

    median_initial = median_image(imgs_raw)
    imgs_med_sub_initial = imgs_raw.astype(np.float32) - median_initial
    mean_pos_arr = np.mean(np.maximum(imgs_med_sub_initial, 0), axis=(1, 2))
    mean_neg_arr = np.mean(np.minimum(imgs_med_sub_initial, 0), axis=(1, 2))
    pos_neg_thresholds = np.asarray([np.percentile(mean_pos_arr, percentile),
                                     np.percentile(mean_neg_arr, 100 - percentile)])
    imgs_after_filter = imgs_raw[
        np.logical_and(mean_pos_arr <= pos_neg_thresholds[0],
                       mean_neg_arr >= pos_neg_thresholds[1]),
    ]
    subtract_mask = median_image(imgs_after_filter)

    if np.max(np.abs(subtract_mask)) > oversaturation_thresh / 6:
        print("Warning! The calculated median mask might have some major outliers!")

    return subtract_mask, pos_neg_thresholds


def median_image(I):
    """
    Receive an array or stack of images and return an image of each pixel's median along the stack
    This calculates the median over each row separately because otherwise there are severe memory issues

    Args:
        I (ndarray): N x height x width stack of images
    Returns:
        median_image (ndarray): height x width stackwise median

    """
    I = np.asarray(I)
    shape = I.shape
    if len(shape) <= 2:
        median_image = I
    else:
        N_images, Ny, Nx = shape
        # Do a loop over rows instead of fully vectorized because otherwise this causes a serious memory problem!!!
        median_image = np.zeros(shape[1:])
        for i in range(Ny):
            median_image[i] = np.median(I[:, i], axis=0)
    return median_image
