import h5py, sys


class Dataset:
    def __init__(self, dataset):
        self.dataset = dataset[()]
        # self.dataset = cv.cvtColor(self.dataset, cv.COLOR_BGR2GRAY)
        self.current_pos = 0
        self.num_examples = len(dataset)

    def next_batch(self, batch_size):
        if self.current_pos + batch_size > self.num_examples:
            self.current_pos = 0
        x = self.dataset[self.current_pos : self.current_pos+batch_size]
        self.current_pos += batch_size
        # return (x, None)
        return (x, None)
        # return (np.reshape(x, (batch_size, 64*64*3)), None)


class Floorplans:
    def __init__(self, root_dir='/mnt/research/datasets/floorplans/'):
        # self.test = Dataset(os.path.join(root_dir, 'test_set.txt'))
        # self.train = Dataset(os.path.join(root_dir, 'train_set.txt'))
        # self.validation = Dataset(os.path.join(root_dir, 'validation_set.txt'))
        self.file = h5py.File("/mnt/research/projects/hem/datasets/floorplan_64_float32.hdf5", 'r')
        sys.stdout.write("Loading test...")
        sys.stdout.flush()
        self.test = Dataset(self.file['test/images'])
        sys.stdout.write("done!\n\rLoading train...")
        sys.stdout.flush()
        self.train = Dataset(self.file['train/images'])
        sys.stdout.write("done!\n\rLoading validation...")
        sys.stdout.flush()
        self.validation = Dataset(self.file['validation/images'])
        sys.stdout.write("done!\n\r")
        sys.stdout.flush()

        print(self.train, self.test, self.validation)
