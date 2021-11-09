import time

from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer_training import Visualizer


if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.

    print(f'The length of training set: {len(dataset)}')

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations

    model.zero_grad_and_step()

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()              # reset the visualizer:

        epoch_losses = None

        # model training
        model.train()

        lr = model.get_learning_rate()
        visualizer.plot_lr(epoch, lr)

        display_iter = -1
        print_iter = -1

        for i, data in enumerate(dataset):  # inner loop within one epoch
            cur_data_size = data['A'].size()[0]
            # print(f'Current batch length: {cur_data.size()}')
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += cur_data_size
            epoch_iter += cur_data_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights
            model.cal_test_loss()

            if total_iters % opt.display_freq == 0 or (total_iters-display_iter) > opt.display_freq:   # display images
                display_iter = total_iters
                save_result = True
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            losses = model.get_current_losses()
            if epoch_losses is None:
                epoch_losses = losses
            else:
                for loss_name in epoch_losses.keys():
                    epoch_losses[loss_name] += losses[loss_name] * cur_data_size

            if total_iters % opt.print_freq == 0 or (total_iters - print_iter) > opt.print_freq:    # print training losses and save logging information to the disk
                print_iter = total_iters
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, losses, epoch_iter, t_comp, t_data)

            # plot loss curve on TensorBoard
            visualizer.plot_current_losses(epoch, losses, cur_iter=total_iters)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()

        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        for loss_name in epoch_losses.keys():
            epoch_losses[loss_name] /= dataset_size
        # plot and print mean loss of current epoch
        visualizer.plot_current_losses(epoch, epoch_losses, is_epoch_loss=True)
        visualizer.print_current_losses(epoch, epoch_losses, is_epoch_loss=True)

        model.update_learning_rate(epoch+1)                     # update learning rates at the end of every epoch.
        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
