import numpy as np
import h5py
import sys


class Dataset:
    def __init__(self, dataset):
        self.dataset = dataset[()]
        # self.dataset = cv.cvtColor(self.dataset, cv.COLOR_BGR2GRAY)
        self.current_pos = 0
        self.num_examples = len(dataset)
        self.indexes = range(self.num_examples)


    def shuffle(self):
        x = np.random.rand(self.num_examples)
        idx_map = np.arange(x.shape[0])
        np.random.shuffle(idx_map)
        self.indexes = idx_map
    

    def next_batch(self, batch_size):
        if self.current_pos + batch_size > self.num_examples:
            self.current_pos = 0
        idx = self.indexes[self.current_pos : self.current_pos+batch_size]
        x = self.dataset[idx]
        self.current_pos += batch_size
        return (x, None)

    @property
    def images(self):
        return self.dataset

    @property
    def labels(self):
        return None


class Floorplans:
    def __init__(self, root_dir='/mnt/research/datasets/floorplans/'):
        # self.test = Dataset(os.path.join(root_dir, 'test_set.txt'))
        # self.train = Dataset(os.path.join(root_dir, 'train_set.txt'))
        # self.validation = Dataset(os.path.join(root_dir, 'validation_set.txt'))
        
        self.file = h5py.File("/mnt/research/projects/hem/datasets/floorplan_64_float32.hdf5", 'r')
        self.test = Dataset(self.file['test/images'])
        self.train = Dataset(self.file['train/images'])
        self.validation = Dataset(self.file['validation/images'])
