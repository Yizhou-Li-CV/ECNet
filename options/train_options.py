from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # visdom and HTML visualization parameters
        parser.add_argument('--display_freq', type=int, default=4000, help='frequency of saving training result images')
        parser.add_argument('--print_freq', type=int, default=1000, help='frequency of showing training results on console')
        # network saving and loading parameters
        parser.add_argument('--save_latest_freq', type=int, default=10000, help='frequency of saving the latest model')
        parser.add_argument('--save_epoch_freq', type=int, default=1, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--save_by_iter', action='store_true', help='whether saves model by iteration')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        # training parameters
        parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs')
        parser.add_argument('--n_epochs_decay', type=int, default=0, help='number of epochs to linearly decay learning rate to zero')
        parser.add_argument('--beta1', type=float, default=0.9, help='beta term of adam')
        parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate for adam')
        parser.add_argument('--lr_policy', type=str, default='multistep', help='learning rate policy. [linear | step | multistep | plateau ]')
        parser.add_argument('--lr_decay_iters', type=int, default=25, help='multiply by a gamma every lr_decay_iters iterations')
        parser.add_argument('--weight_decay', type=float, default=0., help='weight decay for optimizer')
        self.isTrain = True
        return parser
