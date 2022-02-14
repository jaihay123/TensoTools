import json


class JSONData(object):

    def __init__(self, file_path):
        self.file_path = file_path
        self.json_file = None
        self.load_file()

    def load_file(self):
        self.json_file = json.load(open(self.file_path))

    def get_data(self, data):
        return self.json_file[data]

    def get_child_data(self, child, data):
        return child[data]
