# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py
import random
import os
import random
import re
import glob
import json
from pathlib import Path


import numpy as np
import torch
import matplotlib.pyplot as plt

def increment_path(path, exist_ok=False):
    """ Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.
    Args:
        path (str or pathlib.Path): f"{model_dir}/{args.name}".
        exist_ok (bool): whether increment path (increment if False).
    """
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{n}"


def show_image(train, row=5, shuffle=True):
    '''
    Helper function to show image
    :param train: datasets that you want to show
    :param row : rows that you want to display
    :param shuffle : shuffle datset
    '''

    fig = plt.figure(figsize=(8,8*(row//2)), dpi=150)

    if shuffle:
        img_arr = np.random.choice(len(train)-1, row, replace=False)
    else:
        img_arr = [i for i in range(row)]
    
    idx = 1

    for r in range(1,row+1):
        img, mask, infos = train[img_arr[r-1]]
        if torch.is_tensor(img):
            img = img.permute(1,2,0).numpy()
            mask = mask.numpy()
        ax = fig.add_subplot(row, 2, idx)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(img)
        ax.set_title(infos['file_name'], fontsize=7)
        idx += 1

        ax = fig.add_subplot(row, 2, idx)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(mask)
        ax.set_title('classes : ' + ', '.join(list(map(lambda x:str(int(x)), np.unique(mask)))),
         fontsize=7)
        idx += 1


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark=True

    np.random.seed(seed)
    random.seed(seed)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def save_model(model, saved_dir, file_name='fcn8s_best_model(pretrained).pt', save_limit=10):
    check_point = {'net': model.state_dict()}
    output_path = os.path.join(saved_dir, file_name)
    file_list = os.listdir(saved_dir)
    for fl in sorted(file_list, key=lambda x:int(x.split('_')[1]))[:-save_limit-1]:
        os.remove(os.path.join(saved_dir, fl))


    torch.save(model.state_dict(), output_path)


def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist


def label_accuracy_score(label_trues, label_preds, n_class):
    """Returns accuracy score evaluation result.
      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    with np.errstate(divide='ignore', invalid='ignore'):
        acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    with np.errstate(divide='ignore', invalid='ignore'):
        iu = np.diag(hist) / (
            hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)
        )
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc