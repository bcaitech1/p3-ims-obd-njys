import argparse
import os
from importlib import import_module
from pathlib import Path
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
from pycocotools.coco import COCO
import albumentations as A
from albumentations.pytorch import ToTensorV2

from utils import *
from dataset import *

import wandb

def get_train_transform():
    return A.Compose([
                        ToTensorV2()
                        ])

def get_val_transform():
    return A.Compose([
                        ToTensorV2()
                        ])

def get_test_transform():
    return A.Compose([
                        ToTensorV2()
                        ])



def train(args):
    wandb.init(project='Pstage3', name=f'{args.name}')
    wandb.config.update(args)

    seed_everything(args.seed)
    args.name = args.name.replace(' ','_')
    saved_dir = f'saved/{args.name}_{args.model}'

    # -- settings
    device = "cuda" if torch.cuda.is_available() else "cpu" 


    # -- transform
    train_transform = get_train_transform()
    val_transform = get_val_transform()
    test_transform = get_test_transform()

    # -- dataset
    train_dataset, train_loader = get_DataLoader(args.dataset, 'train', transform=train_transform,
                                                batch_size=args.batch_size, shuffle=args.shuffle,
                                                num_workers=args.num_workers)
    
    val_dataset, val_loader = get_DataLoader(args.dataset, 'val', transform=val_transform,
                                            batch_size=args.valid_batch_size, shuffle=args.shuffle,
                                            num_workers=1)
    
    test_dataset, test_loader = get_DataLoader(args.dataset, 'test', transform=test_transform,
                                            batch_size=args.valid_batch_size, shuffle=False,
                                            num_workers=1)
    

    num_classes = args.num_classes

        # -- model
    model_module = getattr(import_module("model"), args.model)  # default: BaseModel
    
    model = model_module(
        num_classes=num_classes
    ).to(device)
    # model = torch.nn.DataParallel(model)

        # -- loss & metric
    criterion = nn.CrossEntropyLoss()
    opt_module = getattr(import_module("torch.optim"), args.optimizer)  # default: SGD
    optimizer = opt_module(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=5e-4
    )
    scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.5)


    print('Start training..')
    best_loss = np.Inf
    for epoch in range(args.epochs):
        model.train()
        for step, (images, masks, _) in enumerate(train_loader):
            images = torch.stack(images)       # (batch, channel, height, width)
            masks = torch.stack(masks).long()  # (batch, channel, height, width)
            
            # gpu 연산을 위해 device 할당
            images, masks = images.to(device), masks.to(device)
            
            # inference
            outputs = model(images)
            
            # loss 계산 (cross entropy loss)
            loss = criterion(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # step 주기에 따른 loss 출력
            if (step + 1) % 25 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                    epoch+1, args.epochs, step+1, len(train_loader), loss.item()))
        
        scheduler.step()
        # validation 주기에 따른 loss 출력 및 best model 저장
        if (epoch + 1) % args.val_every == 0:
            avrg_loss = validation(epoch + 1, model, val_loader, criterion, device)
            if avrg_loss < best_loss:
                if not os.path.isdir(saved_dir):
                    os.mkdir(saved_dir)

                print('Best performance at epoch: {}'.format(epoch + 1))
                print('Save model in', saved_dir)
                best_loss = avrg_loss
                save_model(model, saved_dir=saved_dir, file_name = f'epoch_{epoch}_loss_{best_loss}.pth')
    

    submission = pd.read_csv('./submission/sample_submission.csv', index_col=None)
    file_names, preds = test(model, test_loader, device)
    # PredictionString 대입
    for file_name, string in zip(file_names, preds):
        submission = submission.append({"image_id" : file_name, "PredictionString" : ' '.join(str(e) for e in string.tolist())}, 
                                    ignore_index=True)

    # submission.csv로 저장
    _, saved_dir = saved_dir.split('/')
    submission.to_csv(f"./submission/{saved_dir}_loss_{best_loss}.csv", index=False)


def validation(epoch, model, data_loader, criterion, device):
    print('Start validation #{}'.format(epoch))
    model.eval()
    with torch.no_grad():
        total_loss = 0
        cnt = 0
        mIoU_list = []
        for step, (images, masks, _) in enumerate(data_loader):
            
            images = torch.stack(images)       # (batch, channel, height, width)
            masks = torch.stack(masks).long()  # (batch, channel, height, width)
            images, masks = images.to(device), masks.to(device)            

            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss
            cnt += 1
            
            outputs = torch.argmax(outputs.squeeze(), dim=1).detach().cpu().numpy()

            mIoU = label_accuracy_score(masks.detach().cpu().numpy(), outputs, n_class=12)[2]
            mIoU_list.append(mIoU)
            
        avrg_loss = total_loss / cnt
        print('Validation #{}  Average Loss: {:.4f}, mIoU: {:.4f}'.format(epoch, avrg_loss, np.mean(mIoU_list)))
    return avrg_loss


def test(model, test_loader, device):
    size = 256
    transform = A.Compose([A.Resize(256, 256)])
    print('Start prediction.')
    model.eval()
    
    file_name_list = []
    preds_array = np.empty((0, size*size), dtype=np.long)
    
    with torch.no_grad():
        for step, (imgs, image_infos) in enumerate(test_loader):

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=40, help='number of epochs to train (default: 1)')
    parser.add_argument('--shuffle', type=bool, default=True, help='shuffle')
    parser.add_argument('--num_workers', type=int, default=4, help='num_workers')
    parser.add_argument('--dataset', type=str, default='../input/data', help='dataset directory')
    parser.add_argument('--num_classes', type=int, default=12, help='number of classes')
    parser.add_argument('--batch_size', type=int, default=8, help='input batch size for training (default: 64)')
    parser.add_argument('--valid_batch_size', type=int, default=128, help='input batch size for validing (default: 1000)')
    parser.add_argument('--val_every', type=int, default=1, help='validation every {val_every}')
    parser.add_argument('--model', type=str, default='FCN8s', help='model type (default: FCN8s)')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer type (default: Adam)')
    parser.add_argument('--lr', type=float, default=3e-4, help='learning rate (default: 3e-4)')
    parser.add_argument('--lr_decay_step', type=int, default=10, help='learning rate scheduler deacy step (default: 20)')
    parser.add_argument('--name', default='Baseline Code', help='model save at')

    # Container environment
    args = parser.parse_args()
    print(args)

    train(args)