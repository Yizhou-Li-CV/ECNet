import numpy as np
import os
import ntpath
from . import util
import cv2


def save_images(result_dir, visuals, image_path):

    image_dir = result_dir.get_image_dir()
    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]

    for label, im_data in visuals.items():
        if 'lcn' not in label:
            im = util.tensor2im(im_data, normalized=True)
        else:
            im = util.tensor2im(im_data, normalized=False)
        im = im.astype(np.uint8)

        image_name = '%s_%s.png' % (name, label)
        save_path = os.path.join(image_dir, image_name)
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, im)


