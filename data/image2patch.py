import cv2
import numpy as np
import os


def Im2Patch(img, win=96, stride=96):
    k = 0
    endc = img.shape[0]
    endw = img.shape[1]
    endh = img.shape[2]
    patch = img[:, 0:endw - win + 0 + 1:stride, 0:endh - win + 0 + 1:stride]
    TotalPatNum = patch.shape[1] * patch.shape[2]
    print(TotalPatNum)
    Y = np.zeros([endc, win * win, TotalPatNum], np.float32)

    for i in range(win):
        for j in range(win):
            # print(i * j)
            patch = img[:, i:endw - win + i + 1:stride, j:endh - win + j + 1:stride]
            # print(patch.shape)
            Y[:, k, :] = np.array(patch[:]).reshape(endc, TotalPatNum)
            k = k + 1
    return Y.reshape([endc, win, win, TotalPatNum])


def save_patches(tgt_path, patches, start_idx, prefix):
    c, h, w, n = patches.shape
    for i in range(n):
        patch = patches[:, :, :, i]
        patch = patch.transpose(1, 2, 0)
        save_path = os.path.join(tgt_path, f'{prefix}-{start_idx + i}.png')
        cv2.imwrite(save_path, patch)
    return n


def generate_patch_dataset(src_path, tgt_path, prefix, patch_size=96, horizontal_flip=False):
    idx_counter = 0

    file_names = sorted(os.listdir(src_path))
    print(file_names)

    if not os.path.exists(tgt_path):
        os.makedirs(tgt_path)

    for name in file_names:
        img = cv2.imread(os.path.join(src_path, name))

        patches = Im2Patch(img.transpose(2, 0, 1), win=patch_size, stride=patch_size)
        n = save_patches(tgt_path, patches, idx_counter, prefix)
        idx_counter += n

        print(f'current idx: {idx_counter}')

        if horizontal_flip:
            img_hp = cv2.flip(img, 1)
            patches = Im2Patch(img_hp.transpose(2, 0, 1), win=patch_size, stride=patch_size)
            n = save_patches(tgt_path, patches, idx_counter, prefix)
            idx_counter += n
            print(f'current idx: {idx_counter}')


if __name__ == '__main__':

    suffixes = ['rain', 'norain']

    for suffix in suffixes:
        src_path = './SPA-Data_6385/train/rgb_reconstruction/%s' % suffix
        tgt_path = './SPA-Data_6385_patch/train/rgb_reconstruction/%s' % suffix
        horizontal_flip = False
        prefix = src_path.split('/')[-1]
        patch_size = 96
        generate_patch_dataset(src_path, tgt_path, prefix, patch_size, horizontal_flip)