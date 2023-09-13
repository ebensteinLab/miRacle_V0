import numpy as np


def blobs_to_crops(images, blobs_df, crop_dims):
    """
    Extract crops from the images based on the blobs in blobs_df. Remove any crops too close to the edges.
    Args:
        images (ndarray): N x height x width stack of images from which crops are taken
        blobs_df (pandas df): len N_blobs df of blobs from the images
        crop_dims (2x2 ndarray): dimensions of a normal crop to take, in the format of: [[-up, +down],[-left, +right]]
    Returns:
        out_inds (ndarray of ints): len M<N_blobs array with the indices (up to N_blobs) of crops kept (i.e. not too
        close to the image boundaries).
        crop_out (ndarray): M x crop_height x crop_width array of normal sized crops
        out_LC (ndarray): M x crop_height*2 x crop_width*2 array of large sized crops (with zero padding at image edges)
        out_XLC (ndarray): M x crop_height*10 x crop_width*10 array of XL sized crops (with zero padding at edges)
    """

    L_factor = 2
    XL_factor = 10
    N_blobs = len(blobs_df)
    y_arr = blobs_df.y.values
    x_arr = blobs_df.x.values
    frame_arr = blobs_df.frame.values

    padded_mask = np.zeros(N_blobs, dtype=bool)
    crop_arr = np.zeros([N_blobs, int(np.diff(crop_dims[0])), int(np.diff(crop_dims[1]))], dtype=np.uint16)

    padded_mask_LC = np.zeros(N_blobs, dtype=bool)
    crop_arr_LC = np.zeros([N_blobs, 2 * int(np.diff(crop_dims[0])), L_factor * int(np.diff(crop_dims[1]))],
                           dtype=np.uint16)
    padded_mask_XLC = np.zeros(N_blobs, dtype=bool)
    crop_arr_XLC = np.zeros([N_blobs, 10 * int(np.diff(crop_dims[0])), XL_factor * int(np.diff(crop_dims[1]))],
                            dtype=np.uint16)

    for i in range(N_blobs):
        crop_arr[i], padded_mask[i] = crop_blob(images[frame_arr[i]], y_arr[i], x_arr[i], crop_dims)
        crop_arr_LC[i], padded_mask_LC[i] = crop_blob(images[frame_arr[i]], y_arr[i], x_arr[i], L_factor * crop_dims)
        crop_arr_XLC[i], padded_mask_XLC[i] = crop_blob(images[frame_arr[i]], y_arr[i], x_arr[i],
                                                        XL_factor * crop_dims)

    crop_out = crop_arr[padded_mask]
    # we do not return the padded_mask_LC and padded_mask_XLC because they are used just for visual purposes on
    # V_TIMDER so it is ok if they exceed the image boundary (there is a padding)
    out_inds = np.where(padded_mask)[0]
    out_LC = crop_arr_LC[padded_mask]
    out_XLC = crop_arr_XLC[padded_mask]
    return out_inds, crop_out, out_LC, out_XLC


def crop_blob(img, blob_y, blob_x, crop_dims, padsize=200):
    """
    Extract a single crop from an image based on the coordinates and dimensions.
    Add a zero padding if the crop exceeds image boundaries.
    Args:
        img (ndarray): height x width single frame from the images stack
        blob_y (float): y coordinate of the top blob around which the crop is taken
        blob_x (float): x coordinate of the top blob around which the crop is taken
        crop_dims (2x2 ndarray): dimensions of a normal crop to take, in the format of: [[-up, +down],[-left, +right]]
        padsize (int): maximum size for padding, should be more than 10 * max(abs(crop_dims))
    Returns:
        crop (ndarray): crop_height x crop_width crop
        padded (bool): True if the crop coordinates are within the image, False if they exceed the boundary and the
        padding is needed.
    """
    N_y, N_x = img.shape

    # this is supossed to be the top blob!
    y = int(np.round(blob_y))
    x = int(np.round(blob_x))
    y_min = y + crop_dims[0][0]
    y_max = y + crop_dims[0][1]
    x_min = x + crop_dims[1][0]
    x_max = x + crop_dims[1][1]

    if ((x_min < 0) | (y_min < 0)) | ((x_max > N_x - 1) | (y_max > N_y - 1)):
        padded_image = np.pad(img, padsize)
        crop = padded_image[(padsize + y_min):(padsize + y_max), (padsize + x_min):(padsize + x_max)]
        padded = False
    else:
        crop = img[y_min:y_max, x_min:x_max]
        padded = True

    return crop, padded
