"""Training and evaluation harness for hem.

This code is responsible for interpreting command line arguments and using them to
instantiate datasets, models, and training scenarios, and to then train the models with the
given parameters. It also handles inference, summary generation, and metric reporting, and
supports checkpoint saving to and resumption from disk.

Datasets and models operate on a dynamic plugin system: their directories are searched for
applicable classes, which can then be instantiated based upon the parameters given.

Command line arguments are parsed on an as-known basis, meaning that the general, multi-purpose
arguments are handled here, and the remaining unknown ones are then passed to the dataset for
parsing, and the remainder of *those* then handed off to the model for parsing. In this way
datasets and models can define their own paramaters that may only be specific to them without
changing the overall training harness/framework.

There are, however, conventions to follow. Datasets must inherit from the DataPlugin class
and models from the ModelPlugin class. Each of these specifies methods that must be overridden
and are expected to be defined in order for the training harness to work. See the respective
directories for examples.
"""

import colored_traceback
colored_traceback.add_hook()
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'     # only log errors
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)   # only log errors
import random
import sys
import time
from tqdm import tqdm
import hem





# TODO add some of these things to the graph or other global state
# (so they can be querried via hem.get_xxxxx() functions, rather than keeping a variable handing aruond)


hem.message('Parsing arguments...')
args = hem.parse_args()
hem.message('Initializing working dir...')
hem.init_working_dir(args)
hem.message('Initializing input pipeline...')
x, handle, iterators = hem.get_dataset_tensors(args)
hem.message('Initializing model...')
model = hem.get_model(args.model)(x, args)
hem.message('Initializing supervisor...')
sv = hem.HemSupervisor(args) # TODO handle enabling profiling here
hem.train(model, iterators, handle, sv, args)
sys.exit(1)


# dataset:
# x
# switch_placeholder
# iterators dict


# TODO: hem.init_profiling...
# profiling (optional)
# requires adding libcupti.so.8.0 (or equivalent) to LD_LIBRARY_PATH.
# (location is /cuda_dir/extras/CUPTI/lib64)
# run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE) if args.profile else None
# run_metadata = tf.RunMetadata() if args.profile else None


