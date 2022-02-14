import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Reshape, concatenate
from keras.layers.core import Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D, Conv2D, MaxPooling2D
from keras.optimizers import SGD
from TensoTools.imageLoader import combine_normalised_images, img_from_normalised
from PIL import Image
import os
from TensoTools.wordTokenizer import GloveModel


class DCGANModel(object):
    model_name = 'dc-gan'

    def __init__(self):
        self.generator = None
        self.discriminator = None
        self.model = None
        self.width = 64
        self.height = 64
        self.channels = 3
        self.text_vectors = 300
        self.config = None
        self.glove_path = './glove.6B'
        self.glove_model = GloveModel()

    @staticmethod
    def configuration_file_path(model_dir_path):
        return os.path.join(model_dir_path, DCGANModel.model_name + '-configuration.npy')

    @staticmethod
    def weights_file_path(model_dir_path, model_type):
        return os.path.join(model_dir_path, DCGANModel.model_name + '-' + model_type + '-weights.h5')

    def create_models(self):
        img_width = self.width // 4
        img_height = self.height // 4

        text_input1 = Input(shape=(self.text_vectors,))
        text_layer1 = Dense(1024)(text_input1)
        generator_layer = Activation('tanh')(text_layer1)
        generator_layer = Dense(128 * img_width * img_height)(generator_layer)
        generator_layer = BatchNormalization()(generator_layer)
        generator_layer = Activation('tanh')(generator_layer)
        generator_layer = Reshape((img_width, img_height, 128),input_shape=(128 * img_width * img_height,))(generator_layer)
        generator_layer = UpSampling2D(size=(2, 2))(generator_layer)
        generator_layer = Conv2D(64, kernel_size=5, padding='same')(generator_layer)
        generator_layer = Activation('tanh')(generator_layer)
        generator_layer = UpSampling2D(size=(2, 2))(generator_layer)
        generator_layer = Conv2D(self.channels, kernel_size=5, padding='same')(generator_layer)
        generator_output = Activation('tanh')(generator_layer)

        self.generator = Model(text_input1, generator_output)

        self.generator.compile(loss='mean_squared_error', optimizer="SGD")

        print('generator: ', self.generator.summary())

        text_input2 = Input(shape=(self.text_vectors,))
        text_layer2 = Dense(1024)(text_input2)
        img_input2 = Input(shape=(self.width, self.height, self.channels))

        img_layer2 = Conv2D(64, kernel_size=(5, 5), padding='same')(
            img_input2)
        img_layer2 = Activation('tanh')(img_layer2)
        img_layer2 = MaxPooling2D(pool_size=(2, 2))(img_layer2)
        img_layer2 = Conv2D(128, kernel_size=5)(img_layer2)
        img_layer2 = Activation('tanh')(img_layer2)
        img_layer2 = MaxPooling2D(pool_size=(2, 2))(img_layer2)
        img_layer2 = Flatten()(img_layer2)
        img_layer2 = Dense(1024)(img_layer2)

        merged = concatenate([img_layer2, text_layer2])

        discriminator_layer = Activation('tanh')(merged)
        discriminator_layer = Dense(1)(discriminator_layer)
        discriminator_output = Activation('sigmoid')(discriminator_layer)

        self.discriminator = Model([img_input2, text_input2], discriminator_output)

        doptim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
        self.discriminator.compile(loss='binary_crossentropy', optimizer=doptim)

        print('discriminator: ', self.discriminator.summary())

        model_output = self.discriminator([self.generator.output, text_input1])
        self.model = Model(text_input1, model_output)

        self.discriminator.trainable = False

        goptim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
        self.model.compile(loss='binary_crossentropy', optimizer=goptim)
        print('generator and discriminator: ', self.model.summary())


    def load_model(self, model_dir_path):
        config_path = DCGANModel.configuration_file_path(model_dir_path)
        self.config = np.load(config_path, allow_pickle=True).item()
        self.width = self.config['img_width']
        self.height = self.config['img_height']
        self.channels = self.config['img_channels']
        self.text_vectors = self.config['text_input_dim']
        self.glove_path = self.config['glove_source_dir_path']
        self.create_models()
        self.glove_model.load(self.glove_path, text_vectors=self.text_vectors)
        self.generator.load_weights(DCGANModel.weights_file_path(model_dir_path, 'generator'))
        self.discriminator.load_weights(DCGANModel.weights_file_path(model_dir_path, 'discriminator'))

    def train_model(self, model_dir_path, image_pairs, epochs=None, batch_size=None, dir_path=None, interval=None):
        if epochs is None:
            epochs = 100

        if batch_size is None:
            batch_size = 10

        if interval is None:
            interval = 20

        self.config = dict()
        self.config['img_width'] = self.width
        self.config['img_height'] = self.height
        self.config['text_input_dim'] = self.text_vectors
        self.config['img_channels'] = self.channels
        self.config['glove_source_dir_path'] = self.glove_path

        config_path = DCGANModel.configuration_file_path(model_dir_path)
        np.save(config_path, self.config)

        text_batch = np.zeros((batch_size, self.text_vectors))
        self.glove_model.load(dir_path=self.glove_path, text_vectors=self.text_vectors)
        self.create_models()

        for epoch in range(epochs):
            batch_count = int(image_pairs.shape[0] / batch_size)
            print()
            print("Number of Batches: ", batch_count)
            for batch_index in range(batch_count):
                image_label_pair_batch = image_pairs[batch_index * batch_size:(batch_index + 1) * batch_size]
                image_batch = []
                for index in range(batch_size):
                    image_label_pair = image_label_pair_batch[index]
                    normalized_img = image_label_pair[0]
                    text = image_label_pair[1]
                    image_batch.append(normalized_img)
                    text_batch[index, :] = self.glove_model.encode_document(text, self.text_vectors)

                image_batch = np.array(image_batch)

                generated_images = self.generator.predict(text_batch, verbose=0)

                if (epoch * batch_size + batch_index) % interval == 0 and dir_path is not None:
                    self.save_images(generated_images, dir_path=dir_path,
                                     epoch=epoch, batch_index=batch_index)

                self.discriminator.trainable = True
                dloss = self.discriminator.train_on_batch([np.concatenate((image_batch, generated_images)),
                                                            np.concatenate((text_batch, text_batch))],
                                                           np.array([1] * batch_size + [0] * batch_size))
                print("Epoch: %d " % epoch)
                print("Batch Index: %d " % batch_index)
                print("Discriminator Loss %f " % dloss)
                print()
                self.discriminator.trainable = False
                gloss = self.model.train_on_batch(text_batch, np.array([1] * batch_size))

                if (epoch * batch_size + batch_index) % 10 == 9:
                    self.generator.save_weights(DCGANModel.weights_file_path(model_dir_path, 'generator'), True)
                    self.discriminator.save_weights(DCGANModel.weights_file_path(model_dir_path, 'discriminator'), True)
                    print("Adjusting Weights")

        self.generator.save_weights(DCGANModel.weights_file_path(model_dir_path, 'generator'), True)
        self.discriminator.save_weights(DCGANModel.weights_file_path(model_dir_path, 'discriminator'), True)

    def image_from_text(self, text):
        encoded_text = np.zeros(shape=(1, self.text_vectors))
        encoded_text[0, :] = self.glove_model.encode_document(text)
        generated_images = self.generator.predict(encoded_text, verbose=0)
        generated_image = generated_images[0]
        generated_image = generated_image * 127.5 + 127.5
        return Image.fromarray(generated_image.astype(np.uint8))

    def save_images(self, generated_images, dir_path, epoch, batch_index):
        image = combine_normalised_images(generated_images)
        img_from_normalised(image).save(
        os.path.join(dir_path, DCGANModel.model_name + "-" + str(epoch) + "-" + str(batch_index) + ".png"))