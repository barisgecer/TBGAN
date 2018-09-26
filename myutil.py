import numpy as np
import os
import PIL.Image
import matplotlib
matplotlib.use('Qt4Agg')
from matplotlib import pyplot as pt

def crop_im(img):
    img = img.crop((41, 0, img.size[0] - 42, 377))
    new_img = PIL.Image.new("RGB", (512, 512), (0, 0, 0))
    new_img.paste(img, ((512 - img.size[0]) // 2, (512 - img.size[1]) // 2))
    return new_img


def concat_image(im1, im2):
    if type(im1) is not PIL.Image.Image:
        im1 = PIL.Image.fromarray(im1)
    if type(im2) is not PIL.Image.Image:
        im2 = PIL.Image.fromarray(im2)

    new_im = PIL.Image.new('RGB', (512, 377 * 2))
    new_im.paste(im1.crop((0, 67, im1.size[0], im1.size[1] - 68)), (0, 0))
    new_im.paste(im2.crop((0, 67, im2.size[0], im2.size[1] - 68)), (0, 377))
    return new_im

def rgb2tf(img):
    return np.transpose(np.asarray(img)/127.5-1,(2,0,1))

def tf2rgb(img):
    return (np.clip(np.transpose(img[0][0],(1,2,0))*127.5+127.5,0,255)).astype(np.uint8)

def files_gen(path):
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)):
            yield file

def files(path):
    return list(files_gen(path))