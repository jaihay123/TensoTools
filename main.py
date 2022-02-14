from TensoTools.dataset import Dataset
from TensoTools.trainer import TrainerInstance

if __name__ == "__main__":
    d = Dataset('./data/dataset1', 'captions_train2014.json', 32, 150)
    d.load_dataset(False)

    t = TrainerInstance('./models', 16, 100, d)
    t.train()