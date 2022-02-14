from TensoTools.datasetLoader import get_dataset


class Dataset(object):
    def __init__(self, dataset_path, datafile_path, input_resolution, dataset_size):
        self.dataset_path = dataset_path
        self.datafile_path = datafile_path
        self.input_resolution = input_resolution
        self.dataset_size = dataset_size
        self.dataset = None

    def load_dataset(self, from_web):
        self.dataset = get_dataset(self.dataset_size, self.dataset_path, self.datafile_path, self.input_resolution, from_web)

    def get_dataset(self):
        return self.dataset

    def get_img_size(self):
        return self.input_resolution
