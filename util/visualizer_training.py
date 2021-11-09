import os
import time
from . import util

import tensorflow as tf
from tensorflow import summary
import datetime

import numpy as np


class Visualizer():

    def __init__(self, opt):
        self.opt = opt  # cache the option
        self.name = opt.name
        self.saved = False

        self.checkpoint_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.img_dir = os.path.join(self.checkpoint_dir, 'images')
        print('create checkpoint directory %s...' % self.checkpoint_dir)
        util.mkdirs([self.checkpoint_dir, self.img_dir])
        # create a logging file to store training losses
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

        current_time = datetime.datetime.now()
        time_str = datetime.datetime.strftime(current_time, '%Y%m%d:%H%M%S')

        train_log_dir = f'logs/tensorboard/image_deraining/train/{opt.name}_' + time_str
        print('log dir train:', train_log_dir)

        self.train_summary_writer = summary.create_file_writer(train_log_dir)

    def reset(self):
        """Reset the self.saved status"""
        self.saved = False

    def display_current_results(self, visuals, epoch, save_result):
        if save_result or not self.saved:
            self.saved = True
            # save images to the disk
            for label, image in visuals.items():
                if 'lcn' in label:
                    image_numpy = util.tensor2im(image, normalized=False)
                else:
                    image_numpy = util.tensor2im(image)

                image_numpy = np.uint8(image_numpy)

                img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.png' % (epoch, label))
                util.save_image(image_numpy, img_path)

    def plot_current_losses(self, epoch, losses, cur_iter=None, is_epoch_loss=False):
        with self.train_summary_writer.as_default():
            for loss_name, loss_value in losses.items():
                if is_epoch_loss:
                    tf.summary.scalar(loss_name + '_epoch', loss_value, step=epoch)
                else:
                    tf.summary.scalar(loss_name, loss_value, step=cur_iter)

    def plot_lr(self, epoch, lr):
        with self.train_summary_writer.as_default():
            tf.summary.scalar('learning rate curve', lr, step=epoch)

    def print_current_losses(self, epoch, losses, iters=None, t_comp=None, t_data=None, is_epoch_loss=False):
        if is_epoch_loss:
            message = '(Epoch total loss: %d ' % epoch
        else:
            message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, iters, t_comp, t_data)

        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)

        print(message)  # print the message
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message
