import numpy as np


def flow_stack_oversample(flow_stack, crop_dims):
    """
    This function performs oversampling on flow stacks.
    Adapted from pyCaffe's oversample function
    :param flow_stack:
    :param crop_dims:
    :return:
    """
    im_shape = np.array(flow_stack.shape[1:])
    stack_depth = flow_stack.shape[0]
    crop_dims = np.array(crop_dims)

    h_indices = (0, im_shape[0] - crop_dims[0])
    w_indices = (0, im_shape[1] - crop_dims[1])

    h_center_offset = (im_shape[0] - crop_dims[0])/2
    w_center_offset = (im_shape[1] - crop_dims[1])/2

    crop_ix = np.empty((5, 4), dtype=int)

    cnt = 0
    for i in h_indices:
        for j in w_indices:
            crop_ix[cnt, :] = (i, j, i+crop_dims[0], j+crop_dims[1])
            cnt += 1
    crop_ix[4, :] = [h_center_offset, w_center_offset,
                     h_center_offset+crop_dims[0], w_center_offset+crop_dims[1]]

    crop_ix = np.tile(crop_ix, (2,1))

    crops = np.empty((10, flow_stack.shape[0], crop_dims[0], crop_dims[1]),
                     dtype=flow_stack.dtype)

    for ix in xrange(10):
        cp = crop_ix[ix]
        crops[ix] = flow_stack[:, cp[0]:cp[2], cp[1]:cp[3]]
    crops[5:] = crops[5:, :, :, ::-1]
    crops[5:, range(0, stack_depth, 2), ...] = 255 - crops[5:, range(0, stack_depth, 2), ...]
    return crops


def rgb_oversample(image, crop_dims):
    """
    Crop images into the four corners, center, and their mirrored versions.
    Adapted from Caffe
    Parameters
    ----------
    image : (H x W x K) ndarray
    crop_dims : (height, width) tuple for the crops.
    Returns
    -------
    crops : (10 x H x W x K) ndarray of crops.
    """
    # Dimensions and center.
    im_shape = np.array(image.shape)
    crop_dims = np.array(crop_dims)
    im_center = im_shape[:2] / 2.0

    # Make crop coordinates
    h_indices = (0, im_shape[0] - crop_dims[0])
    w_indices = (0, im_shape[1] - crop_dims[1])
    crops_ix = np.empty((5, 4), dtype=int)
    curr = 0
    for i in h_indices:
        for j in w_indices:
            crops_ix[curr] = (i, j, i + crop_dims[0], j + crop_dims[1])
            curr += 1
    crops_ix[4] = np.tile(im_center, (1, 2)) + np.concatenate([
        -crop_dims / 2.0,
         crop_dims / 2.0
    ])
    crops_ix = np.tile(crops_ix, (2, 1))

    # Extract crops
    crops = np.empty((10 , crop_dims[0], crop_dims[1],
                      im_shape[-1]), dtype=np.float32)

    ix = 0
    for crop in crops_ix:
        crops[ix] = image[crop[0]:crop[2], crop[1]:crop[3], :]
        ix += 1
    crops[ix-5:ix] = crops[ix-5:ix, :, ::-1, :]  # flip for mirrors
    return crops


def rgb_to_parrots(frame, oversample=True, mean_val=None, crop_size=None):
    """
    Pre-process the rgb frame for Parrots input
    """
    if mean_val is None:
        mean_val = [104, 117, 123]
    if not oversample:
        ret_frame = (frame - mean_val).transpose((2, 0, 1))
        return ret_frame[None, ...]
    else:
        crops = rgb_oversample(frame, crop_size) - mean_val
        ret_frames = crops.transpose((0, 3, 1, 2))
        return ret_frames


def fast_list2arr(data, offset=None, dtype=None):
    """
    Convert a list of numpy arrays with the same size to a large numpy array.
    This is way more efficient than directly using numpy.array()
    See
        https://github.com/obspy/obspy/wiki/Known-Python-Issues
    :param data: [numpy.array]
    :param offset: array to be subtracted from the each array.
    :param dtype: data type
    :return: numpy.array
    """
    num = len(data)
    out_data = np.empty((num,)+data[0].shape, dtype=dtype if dtype else data[0].dtype)
    for i in xrange(num):
        out_data[i] = data[i] - offset if offset else data[i]
    return out_data

