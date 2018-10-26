import cv2
import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from keras.preprocessing.image import array_to_img
import keras.backend as K
IMG_WIDTH = 512
from PIL import Image, ImageFilter, ImageEnhance
try:
    from PIL import Image as pil_image
except ImportError:
    pil_image = None



def get_more_images(imgs):
    more_images = []
    vert_flip_imgs = []
    hori_flip_imgs = []

    for i in range(0, imgs.shape[0]):
        a = imgs[i, :, :, 0]
        b = imgs[i, :, :, 1]
        c = imgs[i, :, :, 2]

        av = cv2.flip(a, 1)
        ah = cv2.flip(a, 0)
        bv = cv2.flip(b, 1)
        bh = cv2.flip(b, 0)
        cv = cv2.flip(c, 1)
        ch = cv2.flip(c, 0)

        vert_flip_imgs.append(np.dstack((av, bv, cv)))
        hori_flip_imgs.append(np.dstack((ah, bh, ch)))

    v = np.array(vert_flip_imgs)
    h = np.array(hori_flip_imgs)

    more_images = np.concatenate((imgs, v, h))

    return more_images


def duplicate_labels(labels):
    more_images = []
    vert_flip_imgs = []
    hori_flip_imgs = []
    for i in range(0, labels.shape[0]):
        a = labels[i, :, :, 0]

        av = cv2.flip(a, 1)
        ah = cv2.flip(a, 0)

        vert_flip_imgs.append(av.reshape(IMG_WIDTH, IMG_WIDTH, 1))
        hori_flip_imgs.append(ah.reshape(IMG_WIDTH, IMG_WIDTH, 1))

    v = np.array(vert_flip_imgs)
    h = np.array(hori_flip_imgs)
    duplicate_labels = np.concatenate((labels, v, h))
    return duplicate_labels


def img_to_array(img, data_format=None):
    """Converts a PIL Image instance to a Numpy array.
    # Arguments
        img: PIL Image instance.
        data_format: Image data format.
    # Returns
        A 3D Numpy array.
    # Raises
        ValueError: if invalid `img` or `data_format` is passed.
    """
    if data_format is None:
        data_format = K.image_data_format()
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Unknown data_format: ', data_format)
    # Numpy array x has format (height, width, channel)
    # or (channel, height, width)
    # but original PIL image has format (width, height, channel)
    x = np.asarray(img, dtype=K.floatx())
    if len(x.shape) == 3:
        if data_format == 'channels_first':
            x = x.transpose(2, 0, 1)
    elif len(x.shape) == 2:
        if data_format == 'channels_first':
            x = x.reshape((1, x.shape[0], x.shape[1]))
        else:
            x = x.reshape((x.shape[0], x.shape[1], 1))
    else:
        raise ValueError('Unsupported image shape: ', x.shape)
    return x


def elastic_transform_RGB(image, alpha, sigma, data_format, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
       Works only for "channel_last" data format,
       For original characters size 60-100 pixels use parameters [alpha,sigma]=[4,1.9]
    """

    if random_state is None:
        random_state = np.random.RandomState(None)
    if np.amax(image) <= 1:
        image = (image * 255).astype('uint8')  # otherwise brightness scaling wouldn't work
    else:
        image = image.astype('uint8')
    image = array_to_img(image,scale=False)
    # image = array_to_img(image, data_format, scale=False)
    width, height = image.size
    image = image.resize(size=(28, 28))  #
    image = img_to_array(image)
    shape = image[:, :, 0].shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant",
                         cval=0) * alpha  # create array of random numbers from gaussian distribution with mean=0  std=sigma
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant",
                         cval=0) * alpha  # create array of random numbers from gaussian distribution with mean=0  std=sigma

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))  # create array of image pixel indexes
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))  # shift pixel indxes by dx and dy
    R = map_coordinates(image[:, :, 0], indices, order=1, mode='nearest').reshape(
        shape)  # apply pixel value interpolation channel by channel
    G = map_coordinates(image[:, :, 1], indices, order=1, mode='nearest').reshape(shape)
    B = map_coordinates(image[:, :, 2], indices, order=1, mode='nearest').reshape(shape)
    distorted_image = np.stack((R, G, B), axis=2)
    # distorted_image = array_to_img(distorted_image, scale=False)
    distorted_image = array_to_img(distorted_image)
    distorted_image = distorted_image.resize((width, height), resample=pil_image.BICUBIC)
    distorted_image = distorted_image.filter(ImageFilter.DETAIL)
    distorted_image = ImageEnhance.Brightness(distorted_image)
    final_image = distorted_image.enhance(0.5)
    final_image = img_to_array(final_image, data_format)
    return final_image.astype(K.floatx())


