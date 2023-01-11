## [Single Image Deraining Network with Rain Embedding Consistency and Layered LSTM, WACV2022](https://arxiv.org/abs/2111.03615)
### Introduction
Single image deraining is typically addressed as residual learning to predict the rain layer from an input rainy image. 
For this purpose, an encoder-decoder network draws wide attention, where the encoder is required to encode a high-quality 
rain embedding which determines the performance of the subsequent decoding stage to reconstruct the rain layer. 
However, most of existing studies ignore the significance of rain embedding quality, 
thus leading to limited performance with over/under-deraining. In this paper, with our observation of 
the high rain layer reconstruction performance by an rain-to-rain autoencoder, 
we introduce the idea of "Rain Embedding Consistency" by regarding the encoded embedding by the autoencoder 
as an ideal rain embedding and aim at enhancing the deraining performance by improving the consistency between 
the ideal rain embedding and the rain embedding derived by the encoder of the deraining network. 
To achieve this, a Rain Embedding Loss is applied to directly supervise the encoding process, with a 
Rectified Local Contrast Normalization (RLCN) as the guide that effectively extracts the candidate rain pixels. 
We also propose Layered LSTM for recurrent deraining and fine-grained encoder feature refinement 
considering different scales. Qualitative and quantitative experiments demonstrate that 
our proposed method outperforms previous state-of-the-art methods particularly on a real-world dataset.

## Prerequisites
- Python 3.7, PyTorch >= 1.6.0
- Requirements: opencv-python, tensorflow 1.x/2.x (for use of tensorboard)
- Platforms: Ubuntu 16.04, cuda-10.1 & cuDNN v-7.6.5 (higher versions should also work well)

## Dataset Descriptions 
### Synthetic datasets
* Rain100L: 200 training pairs and 100 test pairs
* Rain100H: 1800 training pairs and 100 test pairs
* Rain200L: 1800 training pairs and 200 test pairs
* Rain200H: 1800 training pairs (same as Rain100H) and 200 test pairs
* Rain800: 700 training pairs and 98 test pairs (we drop 2 images from 100 test images as the images are too large)

### Real-world dataset
* SPA-Data: 638492 training pairs, 1000 testing pairs

Rain100H/L can be downloaded from the [[NetDisk]](https://pan.baidu.com/s/1yV4ih7C4Xg0iazqSBB-U1Q) (pwd:uz8h, link borrowed from https://github.com/hongwang01/RCDNet),

Rain200H/L set can be downloaded from the [[NetDisk]](https://pan.baidu.com/s/1SR7yULy0VZ_JZ4Vawqs7gg#list/path=%2F) 
(link is from http://www.icst.pku.edu.cn/struct/Projects/joint_rain_removal.html), 
which are named as *rain_data_[train/test]_[Light/Heavy].gz*,

Rain800 set can be downloaded from the [[GoogleDrive]](https://drive.google.com/drive/folders/0Bw2e6Q0nQQvGbi1xV1Yxd09rY2s?resourcekey=0-dUoT9AJl1q6fXow9t5TcRQ) 
(link is from https://github.com/hezhangsprinter/ID-CGAN), 
or download our selected set with 98 test pairs [[GoogleDrive]](https://drive.google.com/file/d/1G3FqFvKIJiDvoXx4pbTc0A_Ti1X99URz/view?usp=sharing)

SPA-Data set can be downloaded from the [[NetDisk]](https://mycuhk-my.sharepoint.com/personal/1155152065_link_cuhk_edu_hk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2F1155152065%5Flink%5Fcuhk%5Fedu%5Fhk%2FDocuments%2Fdataset%2Freal%5Fworld%5Frain%5Fdataset%5FCVPR19) 
(link is from https://stevewongv.github.io/#pub).

We also provide our randomly selected 1% SPA-Data mentioned in our paper with 6385 training pairs [[GoogleDrive]](https://drive.google.com/file/d/1qDlnJvbiu9wHDU_cdekD406TcEUB7SZ2/view?usp=sharing)
and full test images [[GoogleDrive]](https://drive.google.com/file/d/1Jq2WEjDAx5Qu2riTcMkB65NOieKvbJdZ/view?usp=sharing)
for quick practice.

## Training

*taking training on 1% SPA-Data (6385 training pairs) as an example*:

1.Download our selected 1% SPA-Data  (including training set) from the [[GoogleDrive]](https://drive.google.com/file/d/1qDlnJvbiu9wHDU_cdekD406TcEUB7SZ2/view?usp=sharing) and unzip to ./data.The unzipped file is like:

 "./data/SPA-Data_6385/train/rgb_reconstruction/rain/rain-\*.png"

 "./data/SPA-Data_6385/train/rgb_reconstruction/norain/norain-\*.png"

Note that if using other datasets, please change the file organization as this.

2.Convert the training images into small patches, where generated image patches are located 
at 

 "./data/SPA-Data_6385_patch/train/rgb_reconstruction/rain/rain-\*.png"

 "./data/SPA-Data_6385_patch/train/rgb_reconstruction/norain/norain-\*.png"

```
$ cd ./data
$ python image2patch.py
```

3.Begining training Autoencoder:
```
$ cd ..
$ tensorboard --logdir logs/tensorboard/image_deraining
$ python train.py --dataroot ./data/SPA-Data_6385_patch --name <autoencoder_path> --model autoencoder_train --dataset_mode rain100h --preprocess none --n_epochs 100 --lr_policy multistep --gradient_clipping 5
```
The tensorboard logs can be found at ./logs/, while trained models can be found at ./checkpoints/<autoencoder_path>/

4.Begining training ECNet or ECNet+LL (by changing --netG to ECNet/ECNetLL):
```
$ tensorboard --logdir logs/tensorboard/image_deraining
$ python train.py --dataroot ./data/SPA-Data_6385_patch --name <ecnet_path> --netG ECNetLL --model ecnet_train_test --autoencoder_checkpoint <autoencoder_path> --dataset_mode rain100h --preprocess none --n_epochs 100 --lr_policy multistep --gradient_clipping 5
```
The tensorboard logs can be found at ./logs/, while trained models can be found at ./checkpoints/<ecnet_path>/


To continue the training at a specified epoch (like 30th epoch), please use the command below:
```
$ python train.py --dataroot ./data/SPA-Data_6385_patch --name <ecnet_path> --netG ECNetLL --model ecnet_train_test --autoencoder_checkpoint <autoencoder_path> --dataset_mode rain100h --preprocess none --n_epochs 100 --lr_policy multistep --gradient_clipping 5 --continue_train --epoch_count 31 --epoch 30
```

## Testing
Download full test images of SPA-Data [[GoogleDrive]](https://drive.google.com/file/d/1Jq2WEjDAx5Qu2riTcMkB65NOieKvbJdZ/view?usp=sharing), and unzip to ./data.
The format is like:

 "./data/spadata/val/rgb_reconstruction/rain/rain-\*.png"

 "./data/spadata/val/rgb_reconstruction/norain/norain-\*.png"

```
$ python test.py --dataroot ./data/spadata --num_test 1000 --name <ecnet_path> --netG ECNetLL --model ecnet_train_test --dataset_mode rain100h --preprocess none
```
**Note that: --num_test is used to set the number of images to be tested.*

The PSNR, SSIM and average inference time will be printed, and derained results are saved in the folder "./results/<ecnet_path>/test_latest/images/".

## Pretrained Model
We place the pretrained models of all the used datasets in the ./checkpoints/.
To use the pretrained model for evaluation, please use the command below for SPA-Data with model trained on full training data:

```
$ python test.py --dataroot ./data/spadata --num_test 1000 --name ecnetll_spadata_full --netG ECNetLL --model ecnet_train_test --dataset_mode rain100h --preprocess none
```
or for Rain100H
```
$ python test.py --dataroot ./data/rain100H --num_test 100 --name ecnetll_100H --netG ECNetLL --model ecnet_train_test --dataset_mode rain100h --preprocess none
```
, etc.

Please change the --name to the corresponding folder name in ./checkpoints.

## Performance Evaluation

All PSNR and SSIM results are computed based on Y channel of YCbCr space. The code is located at ./util/standard_derain_metrics.py

## Acknowledgement 
Code framework borrows from [Pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) by [Jun-Yan Zhu](https://github.com/junyanz/). Thanks for sharing !


## Citation

```
@misc{li2021single,
      title={Single Image Deraining Network with Rain Embedding Consistency and Layered LSTM}, 
      author={Yizhou Li and Yusuke Monno and Masatoshi Okutomi},
      year={2021},
      eprint={2111.03615},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
 ```
