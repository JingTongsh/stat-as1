import numpy as np
import random
import os
import os.path as osp
import glob
import shutil
from typing import List, Dict


def read_contents(file: str) -> List[str]:
    f = open(file, 'r')
    # lines = f.readlines()
    # lines = list(filter(('\n').__ne__, lines))  # remove empty lines
    lines = f.read().splitlines()
    f.close()
    return lines


def array_info(a: np.ndarray):
    print('shape {}, min {}, max {}, mean {}, std {}'.format(a.shape, a.min(), a.max(), a.mean(), a.std()))


def get_class_names(file: str) -> Dict[int, str]:
    lines = read_contents(file)
    ret = {}
    for line in lines:
        sp = line.strip().split()
        k = int(sp[0])
        v = sp[1]
        print(k)
        print(v)
        ret[k] = v
    return ret


def convert(file_feat: str, file_id: str, file_label: str, out_root: str):
    """
    Convert .txt data to .npy files.
    """
    # read files
    lines_feat = read_contents(file_feat)
    num_feat = len(lines_feat)
    print(num_feat)
    lines_id = read_contents(file_id)
    num_id = len(lines_id)
    print(num_id)
    lines_label = read_contents(file_label)
    num_label = len(lines_label)
    print(num_label)

    # get class names
    class_names = get_class_names('data/Animals_with_Attributes2/classes.txt')
    
    # prepare directory
    out_dirs = {}
    for k, v in class_names.items():
        out_dirs[k] = osp.join(out_root, f'{k}-{v}')
        os.makedirs(out_dirs[k], exist_ok=True)

    for feat, name, label in zip(lines_feat, lines_id, lines_label):
        feat = list(map(float, feat.strip().split(' ')))
        feat = np.array(feat)
        array_info(feat)
        label = int(label.strip())
        label_name = class_names[label]
        out_file = osp.join(out_dirs[label], name.strip().split('.')[0] + '.npy')
        np.save(out_file, feat)
        print('feature saved to ' + out_file)


def split_train_test(src_root: str, dst_root: str, test: float = 0.4):
    """
    Split data into train set and test set.
    """
    dst_train = osp.join(dst_root, 'train')
    dst_test = osp.join(dst_root, 'test')
    os.makedirs(dst_train, exist_ok=True)
    os.makedirs(dst_test, exist_ok=True)

    for src_label in os.listdir(src_root):
        k = src_label.split('-')[0]
        src_dir = osp.join(src_root, src_label)
        src_files = glob.glob(osp.join(src_dir, '*.npy'))
        print(src_label, len(src_files))

        random.shuffle(src_files)
        num_test = int(len(src_files) * test)

        for src_file in src_files[:num_test]:
            dst_file = osp.join(dst_test, '{}-{}'.format(k, osp.basename(src_file)))
            shutil.copyfile(src_file, dst_file)
            print(f'{src_file} --> {dst_file}')

        for src_file in src_files[num_test:]:
            dst_file = osp.join(dst_train, '{}-{}'.format(k, osp.basename(src_file)))
            shutil.copyfile(src_file, dst_file)
            print(f'{src_file} --> {dst_file}')


def split_val_folds(src_root: str, num_folds: int):
    """
    Split train set for k-fold validation.
    """
    src_files = glob.glob(osp.join(src_root, '*.npy'))
    src_files.sort()
    print(len(src_files))
    dst_folds = {}
    for k in range(1, num_folds + 1):
        dst_folds[k] = osp.join(src_root, 'fold{:0>2d}'.format(k))
        os.makedirs(dst_folds[k], exist_ok=True)
    
    for k, src_file in enumerate(src_files):
        src_base = osp.basename(src_file)
        dst_file = osp.join(dst_folds[k % num_folds + 1], src_base)
        os.rename(src_file, dst_file)
        print(f'{src_file} --> {dst_file}')



def main(task: int):
    if task <= 1:
        convert(
            'data/Animals_with_Attributes2/Features/ResNet101/AwA2-features.txt',
            'data/Animals_with_Attributes2/Features/ResNet101/AwA2-filenames.txt',
            'data/Animals_with_Attributes2/Features/ResNet101/AwA2-labels.txt',
            'data/converted'
        )
    if task <= 2:
        split_train_test('data/converted', 'data/split')
    if task <= 3:
        split_val_folds('data/split/train', 5) 


if __name__ == '__main__':
    # main(3)
    dirs = ['data/split/test'] + glob.glob('data/split/train/*')
    for pt in dirs:
        print(len(os.listdir(pt)))
