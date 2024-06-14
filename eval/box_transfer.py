import numpy as np
import pandas as pd
import os


def box_transfer(box, w, h, scale):
    """
    Transfer the box from the original image to the resized image

    Inputs:
        - box (np.ndarray): box in the original image (x, y, w, h)
        - w (int): width of the original image
        - h (int): height of the original image
        - scale (int): the scale of the resized image

    Returns:
        - box (np.ndarray): box in the resized image
    """
    size = (h, w)
    max_dim = max(size)
    max_ind = size.index(max_dim)
    box = np.array(box)

    # Resizing
    if max_ind == 0:
        # image is heigher
        wpercent = scale / float(size[0])
        hsize = int((float(size[1]) * float(wpercent)))
        desireable_size = (scale, hsize)
        box = box * wpercent
    else:
        # image is wider
        hpercent = scale / float(size[1])
        wsize = int((float(size[0]) * float(hpercent)))
        desireable_size = (wsize, scale)
        box = box * hpercent

    # Padding
    if max_ind == 0:
        # height fixed at scale, pad the width
        pad_size = scale - desireable_size[1]
        left = int(np.floor(pad_size / 2))
        right = int(np.ceil(pad_size / 2))
        top = int(0)
        bottom = int(0)
    else:
        # width fixed at scale, pad the height
        pad_size = scale - desireable_size[0]
        top = int(np.floor(pad_size / 2))
        bottom = int(np.ceil(pad_size / 2))
        left = int(0)
        right = int(0)

    box[0] = int(np.floor(box[0] + left))
    box[1] = int(np.floor(box[1] + top))
    box[2] = int(np.floor(box[2]))
    box[3] = int(np.floor(box[3]))

    return box.astype(np.int32)


def box2mask(box, w, h):
    """
    Transfer the box to mask

    Inputs:
        - box (np.ndarray): box in the resized image (x, y, w, h)
        - w (int): width of the resized image
        - h (int): height of the resized image

    Returns:
        - mask (np.ndarray): mask in the resized image
    """
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[box[1]:box[1]+box[3], box[0]:box[0]+box[2]] = 1
    return mask

