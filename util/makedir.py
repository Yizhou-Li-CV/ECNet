import os


class MakeDir:

    def __init__(self, result_dir, title):

        self.title = title
        self.result_dir = result_dir
        self.img_dir = os.path.join(self.result_dir, 'images')
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)
        if not os.path.exists(self.img_dir):
            os.makedirs(self.img_dir)

    def get_image_dir(self):
        """Return the directory that stores images"""
        return self.img_dir
