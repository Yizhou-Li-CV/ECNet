import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer_test import save_images
from util import makedir

import time


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.phase = 'val'
    opt.num_threads = 1   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    model = create_model(opt)      # create a model given opt.model and other options

    model.setup(opt)               # regular setup: load and print networks; create schedulers

    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    # create a website
    result_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the result directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        result_dir = '{:s}_iter{:d}'.format(result_dir, opt.load_iter)
    print('creating result directory', result_dir)
    result_dir = makedir.MakeDir(result_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))

    test_losses = None

    last_time = time.time()
    time_meter = 0.
    data_count = 0.

    model.eval()

    for i, data in enumerate(dataset):

        data_time = time.time() - last_time
        last_time = time.time()

        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        interval = model.test()           # run inference

        time_meter += interval
        data_count += 1

        model.cal_test_loss()

        losses = model.get_current_losses(in_test=True)
        result_str = f"idx: {i}"
        for k, v in losses.items():
            result_str += f', loss {k} = {v}'
        print(result_str)

        if test_losses is None:
            test_losses = losses
        else:
            for k, v in losses.items():
                test_losses[k] += v

        if i % 5 == 0:
            print('processing (%04d)-th image' % i)

        if not opt.only_metrics:
            visuals = model.get_current_visuals()  # get image results
            img_path = model.get_image_paths()     # get image paths
            save_images(result_dir, visuals, img_path)

        calculating_time = time.time() - last_time
        last_time = time.time()

        print(f"Data time: {data_time}, Calculation time: {calculating_time}")

    for k in test_losses.keys():
        test_losses[k] /= len(dataset)
        print(f'Test loss {k} from all samples: {test_losses[k]}')

    print('The execution time of per image:', time_meter, data_count, time_meter / data_count)
