import numpy as np
from sklearn import linear_model

from utils.dataset import MyDataset


def train(train_set, val_set):
    x_train, y_train = train_set.load_all()
    x_val, y_val = val_set.load_all()
    model = linear_model.LinearRegression().fit(x_train, y_train)
    score = model.score(x_val, y_val)
    print(score)
    return score

def main():
    # test_set = MyDataset('data/split/test')
    score = 0.0
    for val_fold in range(1, 6):
        train_set = MyDataset('data/split/train', 'train', val_fold)
        val_set = MyDataset('data/split/train', 'val', val_fold)
        score += train(train_set, val_set)
    print(score)


if __name__ == '__main__':
    main()
    