import argparse
import os

from importlib import import_module

import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
from tqdm import tqdm
import torch
from pycocotools.coco import COCO
import albumentations as A

from utils import *
from dataset import *
from train import get_test_transform


def test(model, test_loader, device):
    size = 256
    transform = A.Compose([A.Resize(256, 256)])
    print('Start prediction.')
    model.eval()
    
    file_name_list = []
    preds_array = np.empty((0, size*size), dtype=np.long)
    
    with torch.no_grad():
        for step, (imgs, image_infos) in tqdm(enumerate(test_loader)):

            # inference (512 x 512)
            outs = model(torch.stack(imgs).to(device))
            oms = torch.argmax(outs.squeeze(), dim=1).detach().cpu().numpy()
            
            # resize (256 x 256)
            temp_mask = []
            for img, mask in zip(np.stack(imgs), oms):
                transformed = transform(image=img, mask=mask)
                mask = transformed['mask']
                temp_mask.append(mask)

            oms = np.array(temp_mask)
            
            oms = oms.reshape([oms.shape[0], size*size]).astype(int)
            preds_array = np.vstack((preds_array, oms))
            
            file_name_list.append([i['file_name'] for i in image_infos])
    print("End prediction.")
    file_names = [y for x in file_name_list for y in x]
    
    return file_names, preds_array


def main(args):
    
    device = "cuda" if torch.cuda.is_available() else "cpu" 

    test_transform = get_test_transform()

    # -- dataset
    test_dataset, test_loader = get_DataLoader(args.dataset, 'test', transform=test_transform,
                                            batch_size=args.batch_size, shuffle=False,
                                            num_workers=4)

    model_module = getattr(import_module("model"), args.model)  # default: BaseModel
    
    model = model_module(
        encoder_name=args.encoder_name,
        encoder_weights=args.encoder_weights,
        in_channels=args.in_channels,
        classes=args.num_classes
    ).to(device)

    weight_dir = args.weight_dir
    if '.pt' in weight_dir:
        model = load_model(model, weight_dir)
        file_names, preds = test(model, test_loader, device)
        makeSubmission(file_names, preds, weight_dir)
    else:
        for files in os.listdir(weight_dir):
            if os.path.isdir(files): # error handling 
                continue
            path = os.path.join(weight_dir, files)
            model = load_model(model, path)
            file_names, preds = test(model, test_loader, device)
            print(len(file_names), preds.shape)
            makeSubmission(file_names, preds, path)


def makeSubmission(file_names, preds, weight_dir):
    submission = pd.read_csv('./submission/sample_submission.csv', index_col=None)
    for file_name, string in zip(file_names, preds):
        submission = submission.append({"image_id" : file_name, "PredictionString" : ' '.join(str(e) for e in string.tolist())}, 
                                   ignore_index=True)


    # submission.csv로 저장
    _, model_name, file_name = weight_dir.split('/')
    _, epoch, metric, metric_score = file_name.split('_')
    path = os.path.join(f'submission/{model_name}')
    if not os.path.isdir(path):
        os.mkdir(path)
    metric_score = metric_score.replace('.pth','').replace('.pt','')
    if metric == 'mIoU':
        path = f'{path}/epoch_{epoch}_mIoU_{float(metric_score):.5f}.csv'
    elif metric == 'loss':
        path = f'{path}/epoch_{epoch}_loss_{float(metric_score):.5f}.csv'
    else:
        print("model(.pt || .pth) name error!")
        print("check weight_dir : " + weight_dir)
        return
    submission.to_csv(path, index=False)
    print(f'{path} is generated')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='../input/data', help='dataset directory')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--num_classes', type=int, default=12, help='number of classes')
    parser.add_argument('--model', type=str, default='DeepLabV3Plus', help='model type (default: DeepLabV3Plus)')
    parser.add_argument('--encoder_name', type=str, default='senet154', help='model encoder type (default: senet154)')
    parser.add_argument('--encoder_weights', type=str, default='imagenet', help='model pretrain weight type (default: imagenet)')
    parser.add_argument('--in_channels', type=int, default=3, help='number of channels (default: 3)')
    parser.add_argument('--weight_dir', type=str, default='saved/Baseline_Code_DeepLabV3Plus', help='weight directory folder or file to test')

    args = parser.parse_args()
    print(args)

    main(args)