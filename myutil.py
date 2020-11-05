import numpy as np
import os
import PIL.Image
from menpo.image import Image

def crop_im(img):
    """
    Crops an image.

    Args:
        img: (array): write your description
    """
    img = img.crop((41, 0, img.size[0] - 42, 377))
    new_img = PIL.Image.new("RGB", (512, 512), (0, 0, 0))
    new_img.paste(img, ((512 - img.size[0]) // 2, (512 - img.size[1]) // 2))
    return new_img

def crop_im_512(img_377):
    """
    Crops an image.

    Args:
        img_377: (todo): write your description
    """
    img = img_377
    if isinstance(img_377, Image):
        img = img_377.pixels_with_channels_at_back()
    if img.shape[0]==3:
        np.transpose(img, [1, 2, 0])

    img = img[:, 41:img.shape[1] - 42, :]
    img = np.pad(img, ((67, 68),(0, 0) , (0, 0)), 'constant')
    img = np.clip(img,0,1)
    if isinstance(img_377, Image):
        img = Image(np.transpose(img,[2,0,1]))
    return img

def crop_im_377(img_512):
    """
    Crops an image.

    Args:
        img_512: (todo): write your description
    """
    img = img_512
    if isinstance(img_512, Image):
        img = img_512.pixels_with_channels_at_back()
    if img.shape[0]==3:
        img = np.transpose(img, [1, 2, 0])

    img = img[67:img.shape[1] - 68, :, :]
    img = np.pad(img, ((0, 0),(41, 42) , (0, 0)), 'constant')
    img = np.clip(img,0,1)
    img[:, 0:42, :] = np.transpose(np.tile(img[:, 42, :], [42, 1, 1]), [1, 0, 2])
    img[:, 552:, :] = np.transpose(np.tile(img[:, 552, :], [43, 1, 1]), [1, 0, 2])
    if isinstance(img_512, Image):
        img = Image(np.transpose(img,[2,0,1]))
    return img

def concat_image(im1, im2):
    """
    Concatenate antsimage

    Args:
        im1: (array): write your description
        im2: (array): write your description
    """
    if type(im1) is not PIL.Image.Image:
        im1 = PIL.Image.fromarray(im1)
    if type(im2) is not PIL.Image.Image:
        im2 = PIL.Image.fromarray(im2)

    new_im = PIL.Image.new('RGB', (512, 377 * 2))
    new_im.paste(im1.crop((0, 67, im1.size[0], im1.size[1] - 68)), (0, 0))
    new_im.paste(im2.crop((0, 67, im2.size[0], im2.size[1] - 68)), (0, 377))
    return new_im

def rgb2tf(img):
    """
    Convert the rgb image to rgb.

    Args:
        img: (array): write your description
    """
    return np.transpose(np.asarray(img)/127.5-1,(2,0,1))

def tf2rgb(img):
    """
    Convert rgb rgb rgb image.

    Args:
        img: (array): write your description
    """
    return (np.clip(np.transpose(img[0][0],(1,2,0))*127.5+127.5,0,255)).astype(np.uint8)

def files_gen(path):
    """
    Generate a generator that yields files.

    Args:
        path: (str): write your description
    """
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)):
            yield file

def files(path):
    """
    Generate a list of files.

    Args:
        path: (str): write your description
    """
    return list(files_gen(path))