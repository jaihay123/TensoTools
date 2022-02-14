import os


class ModelTemplate(object):
    model_name = 'model-template'

    def __init__(self):
        self.model = None
        self.config = None
        self.data = None

    #Path to save and get configuration
    @staticmethod
    def configuration_file_path(model_dir_path):
        return os.path.join(model_dir_path, ModelTemplate.model_name + '-configuration.npy')

    #Path to save and get weights
    @staticmethod
    def weights_file_path(model_dir_path, model_type):
        return os.path.join(model_dir_path, ModelTemplate.model_name + '-' + model_type + '-weights.h5')

    #def create_models(self):
        #Model layers are defined here

    #def load_model(self, model_dir_path):
        #Existing model config is loaded here

        #config_path = DCGANModel.configuration_file_path(model_dir_path)
        #self.config = np.load(config_path, allow_pickle=True).item()
        #self.data = self.config['data']
        #self.create_models()

    #def train_model(self, model_dir_path):
        #Model is created an trained here

        #self.config = dict()
        #self.config['data'] = self.data
        #config_path = DCGANModel.configuration_file_path(model_dir_path)
        #np.save(config_path, self.config)
        #self.create_models()

#Any model specific methods go here.