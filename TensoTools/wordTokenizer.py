import numpy as np


def load_glove(dir_path=None, text_vectors=None):
    if text_vectors is None:
        text_vectors = 100

    file_path = dir_path + "/glove.6B." + str(text_vectors) + "d.txt"

    _word_encode = {}

    file = open(file_path, mode='rt', encoding='utf8')
    for line in file:
        words = line.strip().split()
        word = words[0]
        embeds = np.array(words[1:], dtype=np.float32)
        _word_encode[word] = embeds
    file.close()

    return _word_encode


class GloveModel(object):

    model_name = 'glove model'

    def __init__(self):
        self.word_encode = None
        self.text_vectors = None

    def load(self, dir_path, text_vectors=None):
        if text_vectors is None:
            text_vectors = 100
        self.text_vectors = text_vectors
        self.word_encode = load_glove(dir_path, text_vectors)

    def encode_word(self, word):
        w = word.lower()
        if w in self.word_encode:
            return self.word_encode[w]
        else:
            return np.zeros(shape=(self.text_vectors,))

    def encode_document(self, doc, max_doc_length=None):
        words = [w.lower() for w in doc.split(' ')]
        max_len = len(words)
        if max_doc_length is not None:
            max_len = min(len(words), max_doc_length)
        B = np.zeros(shape=(self.text_vectors, max_len))
        A = np.zeros(shape=(self.text_vectors,))
        for j in range(max_len):
            word = words[j]
            try:
                B[:,j] = self.word_encode[word]
            except KeyError:
                pass
        A[:] = np.sum(B, axis=1)
        return A