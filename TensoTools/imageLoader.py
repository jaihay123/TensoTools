import numpy as np
from PIL import Image
from keras_preprocessing.image import img_to_array, load_img
from urllib import request
from io import BytesIO
import math


def combine_normalised_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num) / width))
    shape = generated_images.shape[1:]
    image = np.zeros((height * shape[0], width * shape[1], shape[2]), dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index / width)
        j = index % width
        image[i * shape[0]:(i+1) * shape[0], j * shape[1]:(j+1) * shape[1], :] = img
    return image


def img_from_normalised(normalised_image):
    image = normalised_image * 127.5 + 127.5
    return Image.fromarray(image.astype(np.uint8))


def images_from_web(local_path, img_path, img_size):
    res = request.urlopen(img_path)
    img = Image.open(BytesIO(res.read()))
    img = img.resize((img_size, img_size), Image.NEAREST)
    img.save(local_path + "/" + img_path.split('/')[4])
    print("Saved " + img_path.split('/')[4])


def images_from_file(local_path, img_size):
    image = img_to_array(load_img(local_path, color_mode="rgb", target_size=(img_size, img_size)))
    image = (image.astype(np.float32) / 255) * 2 - 1
    return image
