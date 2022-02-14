from DCGANModel import DCGANModel


class TrainerInstance(object):

    def __init__(self, model_path, batch_size, epochs, dataset):
        self.model_path = model_path
        self.img_size = dataset.get_img_size()
        self.img_channels = 3
        self.batch_size = batch_size
        self.epochs = epochs
        self.dataset = dataset.get_dataset()

    def train(self):
        image_pairs = self.dataset
        gan = DCGANModel()
        gan.width = self.img_size
        gan.height = self.img_size
        gan.channels = self.img_channels
        gan.glove_path = './glove.6B'

        gan.train_model(model_dir_path=self.model_path, image_pairs=image_pairs, dir_path='./data/snapshots',
                        interval=100, batch_size=self.batch_size, epochs=self.epochs)

    def run_model(self, description_text):
        output_path = './data/output/outputImage.png'

        text = description_text
        gan = DCGANModel()
        gan.load_model(self.model_path)

        generated_image = gan.image_from_text(text)
        generated_image.save(output_path)
