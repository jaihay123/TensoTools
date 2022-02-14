import numpy as np
from TensoTools.jsonLoader import JSONData
from TensoTools.imageLoader import images_from_file, images_from_web


def get_dataset(dataset_size, path, data_file, img_size, web):
    images = dict()
    texts = dict()

    json_file = JSONData(path + "/" + data_file)

    for item in json_file.get_data('images'):
        image_id = json_file.get_child_data(item, 'id')
        if not web:
            images[image_id] = path + "/" + json_file.get_child_data(item, 'file_name')
        else:
            images[image_id] = json_file.get_child_data(item, 'coco_url')

    for item in json_file.get_data('annotations'):
        if json_file.get_child_data(item, 'image_id') in texts:
            text_id = json_file.get_child_data(item, 'image_id')
            texts[text_id] = texts[text_id] + " " + json_file.get_child_data(item, 'caption')
        else:
            text_id = json_file.get_child_data(item, 'image_id')
            texts[text_id] = json_file.get_child_data(item, 'caption')

    result = []
    count = 0

    for imageid, text in texts.items():
        if count <= dataset_size:
            if not web:
                result.append([images_from_file(images[imageid], img_size), text])
            else:
                images_from_web(path, images[imageid], img_size)
            count += 1

    if not web:
        return np.array(result)
    else:
        return None


