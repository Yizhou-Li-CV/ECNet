import random
import numpy as np
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
from abc import ABC, abstractmethod

try:
    import accimage
except ImportError:
    accimage = None


class BaseDataset(data.Dataset, ABC):
    """This class is an abstract base class (ABC) for datasets.

    To create a subclass, you need to implement the following four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.
    """

    def __init__(self, opt):
        """Initialize the class; save the options in the class

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.opt = opt
        self.root = opt.dataroot

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        return parser

    @abstractmethod
    def __len__(self):
        """Return the total number of images in the dataset."""
        return 0

    @abstractmethod
    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns:
            a dictionary of data with their names. It ususally contains the data itself and its metadata information.
        """
        pass


def __is_pil_image(img):
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)


def get_params(opt, size):

    w, h = size
    new_h = h
    new_w = w

    if opt.phase == 'train':
        x = random.randint(0, np.maximum(0, new_w - opt.crop_size_w))
        y = random.randint(0, np.maximum(0, new_h - opt.crop_size_h))

        flip = random.random() > 0.5
        vertical_flip = random.random() > 0.5
    else:
        x, y = 0, 0
        flip = False
        vertical_flip = False

    return {'crop_pos': (x, y), 'flip': flip, 'vertical_flip': vertical_flip}


def get_transform(opt, params=None, grayscale=False, convert=True):
    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale(1))

    if 'random_crop' in opt.preprocess:
        transform_list.append(transforms.Lambda(lambda img: __crop_with_size(img, params['crop_pos'], opt.crop_size_w, opt.crop_size_h)))

    if opt.preprocess == 'none':
        transform_list.append(transforms.Lambda(lambda img: __make_power_8(img, base=8)))

    if not opt.no_flip:
        if params['flip']:
            transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))
        if opt.vertical_flip:
            if params['vertical_flip']:
                transform_list.append(transforms.Lambda(lambda img: __vertical_flip(img, params['vertical_flip'])))

    if convert:
        transform_list += [transforms.ToTensor()]
        if grayscale:
            transform_list += [transforms.Normalize((0.5,), (0.5,))]
        else:
            transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

    return transforms.Compose(transform_list)


def __pil_to_ndarray(img):
    return np.asarray(img, dtype=np.float32)


def __make_power_8(img, base=8):
    ow, oh = img.size
    h = int((oh // base) * base)
    w = int((ow // base) * base)
    if h == oh and w == ow:
        return img

    __print_size_warning(ow, oh, w, h)
    return img.crop((0, 0, ow // 8 * 8, oh // 8 * 8))


def __crop_with_size(img, pos, w, h):
    ow, oh = img.size
    x1, y1 = pos
    tw = w
    th = h
    if ow > tw or oh > th:
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img


def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def __vertical_flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_TOP_BOTTOM)
    return img


def __print_size_warning(ow, oh, w, h):
    """Print warning information about image size(only print once)"""
    if not hasattr(__print_size_warning, 'has_printed'):
        print("The image size needs to be a multiple of 8. "
              "The loaded image size was (%d, %d), so it was adjusted to "
              "(%d, %d). This adjustment will be done to all images "
              "whose sizes are not multiples of 8" % (ow, oh, w, h))
        __print_size_warning.has_printed = True
