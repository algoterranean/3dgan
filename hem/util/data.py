import tensorflow as tf
from tensorflow.contrib.data import Iterator
import os
import importlib
import inspect

NCHW = 0
NHWC = 1


def search_for_plugins(plugin_dir='hem/data', plugin_module='hem.data', plugin_name='DataPlugin'):
    valid_plugins = []
    # for each file in the plugins dir, import it and see if
    # there are any classes that inherit from DataPlugin
    files = filter(lambda f: f.endswith('.py') and f not in ['__init__.py', plugin_name+'.py'], os.listdir(plugin_dir))
    for f in files:
        module_name = f[0:-3]
        i = importlib.import_module(plugin_module + '.' + module_name)
        classes = [m for m in inspect.getmembers(i, inspect.isclass)]
        for c in classes:
            if c[1].__module__ == plugin_module + '.' + module_name:
                parents = c[1].__bases__
                # check if this is a valid plugin
                if parents[0].__name__ == plugin_name:
                    valid_plugins.append(c[1])
    loaded_plugins = {}
    for p in valid_plugins:
        loaded_plugins[p.name] = p
    return loaded_plugins


def get_dataset(dataset):
    loaded_plugins = search_for_plugins('hem/data')
    p = loaded_plugins[dataset]
    return p

# args.dataset
# args.dataset_dir
# args.raw_dataset_dir
# args.test_epochs
# args.cache_dir
# args.batch_size
# args.n_gpus
#
# args.n_threads

def get_dataset_tensors(args):
    with tf.device('/cpu:0'), tf.variable_scope('input_pipeline'):
        # TODO move this to hem.init()
        # find all dataset plugins available
        p = get_dataset(args.dataset)
        # ensure that the dataset exists
        if not p.check_prepared_datasets(args.dataset_dir):
            if not p.check_raw_datasets(args.raw_dataset_dir):
                print('Downloading dataset...')
                # TODO: datasets should be able to be marked as non-downloadable
                p.download(args.raw_dataset_dir)
            print('Converting to tfrecord...')
            p.convert_to_tfrecord(args.raw_dataset_dir, args.dataset_dir)

        # load the dataset
        datasets = p.get_datasets(args)
        dataset_iterators = {}
        # tensor to hold which training/eval phase we are in
        handle = tf.placeholder(tf.string, shape=[])
        # add a dataset for train, validation, and testing
        for k, v in datasets.items():
            # skip test set if not needed
            if len(args.test_epochs) == 0 and k == 'test':
                continue
            d = v[0]
            n = sum([1 for r in tf.python_io.tf_record_iterator(v[1])])
            cache_fn = '{}.cache.{}'.format(args.dataset, k)
            d = d.cache(os.path.join(args.cache_dir, cache_fn)) if args.cache_dir else d.cache()
            d = d.repeat()
            d = d.shuffle(buffer_size=args.buffer_size, seed=args.seed)
            d = d.batch(args.batch_size * args.n_gpus)
            x_iterator = d.make_initializable_iterator()
            dataset_iterators[k] = {'x': x_iterator,
                                    'n': n,
                                    'batches': int(n/(args.batch_size * args.n_gpus)),
                                    'handle': x_iterator.string_handle()}
        # feedable dataset that will swap between train/test/val
        iterator = Iterator.from_string_handle(handle,
                                               d.output_types,
                                               d.output_shapes)
        return iterator.get_next(), handle, dataset_iterators




