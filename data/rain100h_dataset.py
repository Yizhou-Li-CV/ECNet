import os.path
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image


class Rain100HDataset(BaseDataset):

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)

        self.dir_A = os.path.join(opt.dataroot, opt.phase, 'rgb_reconstruction', 'rain') # get the rainy image directory
        self.dir_B = os.path.join(opt.dataroot, opt.phase, 'rgb_reconstruction', 'norain')  # get the rain-free image directory
        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))  # get image paths
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))  # get image paths

    def __getitem__(self, index):

        B_path = self.B_paths[index]
        # B_path = self.B_paths[index]
        B_img_name = B_path.split('/')[-1]
        A_img_name = B_img_name.replace('no', '')
        A_path = os.path.join(self.dir_A, A_img_name)
        if not os.path.exists(A_path):
            A_path = os.path.join(self.dir_A, B_img_name)
        A, B = Image.open(A_path).convert('RGB'), Image.open(B_path).convert('RGB')

        # apply the same transform to both A and B
        transform_params = get_params(self.opt, A.size)

        A_transform = get_transform(self.opt, transform_params)
        B_transform = get_transform(self.opt, transform_params)

        A = A_transform(A)
        B = B_transform(B)

        # print('A shape:', A.shape)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.B_paths)
