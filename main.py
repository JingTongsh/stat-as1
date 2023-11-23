import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
import os
import time

from utils.dataset import MyDataset


METHODS = {
    'linear regression': LinearRegression(),
    'logistic regression': LogisticRegression(),
    'lda': LinearDiscriminantAnalysis(),
    'svm': SVC(kernel='linear'),
}


def convert_seconds(s) -> str:
    hours = s // 3600
    minutes = (s % 3600) // 60
    seconds = s % 60
    ans = ''
    if hours > 0:
        ans += f'{hours}h '
    if minutes > 0:
        ans += f'{minutes}min '
    if s >= 60:
        ans += f'{int(seconds)}s'
    else:
        ans += f'{seconds:.2f}s'
    return ans


def train(name: str, train_set: MyDataset, val_set: MyDataset) -> float:
    start = time.time()
    x_train, y_train = train_set.load_all()
    x_val, y_val = val_set.load_all()
    model = METHODS[name].fit(x_train, y_train)
    score = model.score(x_val, y_val)
    duration = time.time() - start
    print('score {}, duration {:.2f}s'.format(score, duration))
    return score


def train_val(method_name: str):
    print(method_name)

    n_fold = len(os.listdir('data/split_data/train'))
    total_score = 0.0
    for val_fold in range(1, n_fold + 1):
        print(f'training: fold {val_fold}/{n_fold}')
        train_set = MyDataset('data/split_data/train', 'train', val_fold)
        val_set = MyDataset('data/split_data/train', 'val', val_fold)
        val_score = train(method_name, train_set, val_set)
        total_score += val_score
    print('avg score', total_score / n_fold)


def train_test(method_name: str):
    print(method_name)
    train_set = MyDataset('data/split_data/train', 'train all')
    test_set = MyDataset('data/split_data/test', 'test')
    score = train(method_name, train_set, test_set)
    print('score', score)


if __name__ == '__main__':
    # for name in METHODS:
    #     train_val(name)
    #     train_test(name)
    train_val('svm')
    train_test('svm')
    # TODO: hyper-parameters
