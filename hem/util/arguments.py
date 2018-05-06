import argparse
import uuid
import multiprocessing
import random
import os

import hem


def parse_args(display=False):
    # parse command line arguments
    ######################################################################
    parser = hem.CustomArgumentParser(description='Autoencoder training harness.',
                                      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                      fromfile_prefix_chars='@',
                                      conflict_handler='resolve',
                                      epilog="""Example: 
                                      python train.py @path/to/config_file 
                                      --dir workspace/model_test 
                                      --lr 0.1""")
    parser._action_groups.pop()

    data_args = parser.add_argument_group('Data')
    optimizer_args = parser.add_argument_group('Optimizer')
    train_args = parser.add_argument_group('Training')
    misc_args = parser.add_argument_group('Miscellaneous')

    # TODO add support for specifying additional directories for data and model plugins

    # misc settings
    add = misc_args.add_argument
    add('--seed',
        type=int,
        help="Useful for debugging. Randomized each execution if not set.")
    add('--n_gpus',
        type=int,
        default=1,
        help="""Number of GPUs to use for simultaneous training. Model will be 
        duplicated on each device and results averaged on CPU.""")
    add('--profile',
        default=False,
        action='store_true',
        help="""Enables runtime metadata collection during training that is 
        viewable in TensorBoard.""")
    add('--check_numerics',
        default=False,
        action='store_true',
        help="""Enables numeric checks for nan/inf in gradients for more detailed 
        error reporting.""")
    add('--model',
        type=lambda s: s.lower(),
        default='fc',
        help="Name of model to train.")

    # training settings
    add = train_args.add_argument
    add('--epochs',
        default='3',
        help="""Number of epochs to train for during this run. Use an integer to
        denote the max number of epochs to train for, or `+n` for an 
        additional n epochs from a saved checkpoint.""")
    add('--batch_size',
        type=int,
        default=256,
        help="Batch size to use, per device.")
    add('--epoch_size',
        type=int,
        default=-1,
        help="""Number of iterations to use per epoch. Defaults to using the 
        entire dataset.""")
    add('--dir',
        type=str,
        default='workspace/{}'.format(uuid.uuid4()),
        help="""Location to store checkpoints, logs, etc. If this location is populated 
        by a previous run then training will be continued from last checkpoint.""")
    add('--max_to_keep',
        type=int,
        default=0,
        help="""Max (most recent) number of saved sessions to keep, once per epoch. 
                    Set to 0 to keep every one.""")
    add('--test_epochs',
        nargs='*',
        default=[],
        type=int,
        help="""List of epochs where the model should be run against the Test dataset.
                    Leave blank to run at the end of training (--epochs argument).""")

    # optimizer settings
    add = optimizer_args.add_argument
    add('--optimizer',
        type=lambda s: s.lower(),
        default='rmsprop',
        help="Optimizer to use during training.")
    add('--lr',
        type=float,
        default=0.001,
        help="Learning rate of optimizer (if supported).")
    add('--loss',
        type=lambda s: s.lower(),
        default='l1',
        help="Loss function used by model during training (if supported).")
    add('--momentum',
        type=float,
        default=0.01,
        help="Momentum value used by optimizer (if supported).")
    add('--decay',
        type=float,
        default=0.9,
        help="Decay value used by optimizer (if supported).")
    add('--centered',
        default=False,
        action='store_true',
        help="Enables centering in RMSProp optimizer.")
    add('--beta1',
        type=float,
        default=0.9,
        help="Value for optimizer's beta_1 (if supported).")
    add('--beta2',
        type=float,
        default=0.999,
        help="Value for optimizer's beta_2 (if supported).")

    # data/pipeline settings
    add = data_args.add_argument
    add('--dataset',
        type=lambda s: s.lower(),
        default='floorplan',
        help="Name of dataset to use.")
    add('--shuffle',
        default=True,
        action='store_true',
        help="""Set this to shuffle the dataset every epoch.""")
    add('--buffer_size',
        type=int,
        default=10000,
        help="""Size of the data buffer.""")
    add('--cache_dir',
        default=None,
        help="""Cache dataset to the directory specified. If not provided, 
        will attempt to cache to memory.""")
    add('--raw_dataset_dir',
        default='/tmp',
        help="Location of raw dataset files, if needed")
    add('--dataset_dir',
        default='datasets',
        help="Location of prepared tfrecord files for the requested dataset.")
    add('--n_threads',
        type=int,
        default=multiprocessing.cpu_count(),
        help="""Number of threads to use for processing datasets.""")

    # parse main/general arguments
    args, leftover_args = parser.parse_known_args()
    # parse dataset-specific arguments
    for k, v in hem.get_dataset(args.dataset).arguments().items():
        parser.add_argument(k, **v)
    args, leftover_args = parser.parse_known_args(leftover_args, namespace=args)

    # parse model-specific arguments
    model = hem.get_model(args.model)
    for k, v in model.arguments().items():
        parser.add_argument(k, **v)
    args, leftover_args = parser.parse_known_args(leftover_args, namespace=args)
    if len(leftover_args) > 0:
        hem.message('WARNING: unknown and unused arguments provided: {}'.format(leftover_args),
                    format_style=hem.WARNING)

    # set seed (useful for debugging purposes)
    if args.seed is None:
        args.seed = os.urandom(4)
    random.seed(args.seed)


    if display:
        for a in vars(args):
            v = getattr(args, a)
            print('    {} = {}'.format(a, v))

    return args

