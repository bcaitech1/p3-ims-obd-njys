import argparse
import os
from importlib import import_module
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
import segmentation_models_pytorch as smp

from pycocotools.coco import COCO
import albumentations as A
from albumentations.pytorch import ToTensorV2

from utils import *
from dataset import *

import wandb
from adamp import AdamP


def get_train_transform(CropSize = 0, height = 224, width = 224):
    if CropSize <= 0:
        return A.Compose([
                        A.Resize(height, width),
                        ToTensorV2()
                        ])
    else:
        return A.Compose([
                        A.CenterCrop(CropSize, CropSize),
                        A.Resize(height, width),
                        ToTensorV2()
                        ])

def get_val_transform(CropSize = 0, height = 224, width = 224):
    if CropSize <= 0:
        return A.Compose([
                        A.Resize(height, width),
                        ToTensorV2()
                        ])
    else:
        return A.Compose([
                        A.CenterCrop(CropSize, CropSize),
                        A.Resize(height, width),
                        ToTensorV2()
                        ])

def get_test_transform(height = 224, width = 224):
    return A.Compose([
                    A.Resize(height, width),
                    ToTensorV2()
                    ])


def rand_bbox(size, lam, half=False)->tuple:
    '''
    랜덤한 bounding box의 좌상단,우하단 좌표 반환

    Args:
        size (tuple): batch의 shape
        lam (float): 자를 비율
        half (bool): 절반으로 자름
    '''

    W = size[2]
    H = size[3]

    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    if half==False:
        bbx1 = np.clip(cx - cut_w // 2, 0, W) 
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
    else:
        bbx1 = 0
        bby1 = 0
        bbx2 = W//2
        bby2 = H

    return bbx1, bby1, bbx2, bby2

def cutmix(image, mask, alpha, half=False):
    '''
    이미지와 마스크 컷믹스

    Args:
        image (tensor): batch 이미지
        mask (tensor): batch 마스크
        alpha (float): Beta Distribution의 alpha 값
    '''
  
    indices = torch.randperm(image.size(0)) # 배치 크기 입력

    lam = np.clip(np.random.beta(alpha, alpha),0.3,0.4)
    bbx1, bby1, bbx2, bby2 = rand_bbox(image.size(), lam, half)
    new_image = image.clone()
    new_mask = mask.clone()
    new_image[:, :, bby1:bby2, bbx1:bbx2] = image[indices, :, bby1:bby2, bbx1:bbx2]
    new_mask[:, bby1:bby2, bbx1:bbx2] = mask[indices, bby1:bby2, bbx1:bbx2]

    return new_image, new_mask

def train(args):
    wandb.init(project='Pstage3', name=f'{args.name}')
    wandb.config.update(args)

    seed_everything(args.seed)
    args.name = args.name.replace(' ','_')
    saved_dir = f'saved/{args.name}_{args.model}'

    # -- settings
    device = "cuda" if torch.cuda.is_available() else "cpu" 


    # -- transform
    train_transform = get_train_transform(CropSize = args.center_crop_size, height = args.image_resize, width = args.image_resize)
    val_transform = get_val_transform(CropSize = args.center_crop_size, height = args.image_resize, width = args.image_resize)
    test_transform = get_test_transform(height = args.image_resize, width = args.image_resize)

    # -- dataset
    train_dataset, train_loader = get_DataLoader(args.dataset, 'train', transform=train_transform,
                                                batch_size=args.batch_size, shuffle=args.shuffle,
                                                num_workers=args.num_workers)
    
    val_dataset, val_loader = get_DataLoader(args.dataset, 'val', transform=val_transform,
                                            batch_size=args.valid_batch_size, shuffle=args.shuffle,
                                            num_workers=4)
    
    test_dataset, test_loader = get_DataLoader(args.dataset, 'test', transform=test_transform,
                                            batch_size=args.valid_batch_size, shuffle=False,
                                            num_workers=1)
    
    print(f'train_data {len(train_dataset)}, val_dataset {len(val_dataset)}, test_dataset {len(test_dataset)} loaded')

    num_classes = args.num_classes

    # -- model
    model_module = getattr(import_module("model"), args.model)  # default: BaseModel
    
    model = model_module(
        encoder_name=args.encoder_name,
        encoder_weights=args.encoder_weights,
        in_channels=args.in_channels,
        classes=num_classes
    ).to(device)
    # model = torch.nn.DataParallel(model)

        # -- loss & metric
    criterion = nn.CrossEntropyLoss()
    if args.optimizer.lower() == 'adamp':
        optimizer = AdamP(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-6)
    else:
        opt_module = getattr(import_module("torch.optim"), args.optimizer)  # default: Adam
        optimizer = opt_module(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            weight_decay=1e-6
        )
    scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.5)
    isCutMix = args.cutmix

    print('Start training..')
    best_loss = np.Inf
    best_mIoU = 0
    for epoch in range(args.epochs):
        model.train()
        mean_loss = 0
        for step, (images, masks, _) in enumerate(train_loader):
            images = torch.stack(images).to(device)       # (batch, channel, height, width)
            masks = torch.stack(masks).long().to(device)  # (batch, channel, height, width)
            
            # 50% 확률 cutmix
            if isCutMix == True:
                mix_decision = np.random.rand()
                if mix_decision < 0.5:
                    # cutmix(data, target, alpha)
                    images, masks = cutmix(images, masks, 1., half=True)
            
            # inference
            outputs = model(images)
            
            # loss 계산 (cross entropy loss)
            loss = criterion(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            mean_loss += loss
            
            # step 주기에 따른 loss 출력
            if (step + 1) % 25 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                    epoch+1, args.epochs, step+1, len(train_loader), loss.item()))

        wandb.log({'train_loss': mean_loss/(step+1)})    
        
        scheduler.step()
        # validation 주기에 따른 loss 출력 및 best model 저장
        if (epoch + 1) % args.val_every == 0:
            avrg_loss, mIoU = validation(epoch + 1, model, val_loader, criterion, device)
            if avrg_loss < best_loss:
                if not os.path.isdir(saved_dir):
                    os.mkdir(saved_dir)

                print('Best performance at epoch: {}'.format(epoch + 1))
                print('Save model in', saved_dir)
                best_loss = avrg_loss
                save_model(model, saved_dir=saved_dir, file_name = f'epoch_{epoch}_loss_{best_loss:.4f}.pth', save_limit=args.save_limit)
            if mIoU > best_mIoU:
                print('[mIoU] Best performance at epoch: {}'.format(epoch + 1))
                print('Save model in', saved_dir)
                best_mIoU = mIoU
                save_model(model, saved_dir, file_name = f'epoch_{epoch}_mIoU_{best_mIoU:.4f}.pth', save_limit=args.save_limit)


def validation(epoch, model, data_loader, criterion, device):
    print('Start validation #{}'.format(epoch))
    model.eval()
    hist = np.zeros((12, 12)) # 12 : num_classes
    with torch.no_grad():
        total_loss = 0
        cnt = 0
        for step, (images, masks, _) in enumerate(data_loader):
            
            images = torch.stack(images).to(device)       # (batch, channel, height, width)
            masks = torch.stack(masks).long().to(device)  # (batch, channel, height, width)

            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss
            cnt += 1
            
            outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            
            hist = add_hist(hist, masks.detach().cpu().numpy(), outputs, n_class=12)
            
        acc, acc_cls, mIoU, fwavacc = label_accuracy_score(hist)    
        
        wandb.log({'val_acc': acc, 'mIoU':mIoU})

        avrg_loss = total_loss / cnt
        print('Validation #{}  Average Loss: {:.4f}, mIoU: {:.4f}, acc : {:.4f}'.format(epoch, avrg_loss, mIoU, acc))

    return avrg_loss, mIoU


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=1, help='number of epochs to train (default: 1)')
    parser.add_argument('--shuffle', type=bool, default=True, help='shuffle')
    parser.add_argument('--num_workers', type=int, default=4, help='num_workers')
    parser.add_argument('--dataset', type=str, default='../input/data', help='dataset directory')
    parser.add_argument('--num_classes', type=int, default=12, help='number of classes')
    parser.add_argument('--batch_size', type=int, default=8, help='input batch size for training (default: 8)')
    parser.add_argument('--valid_batch_size', type=int, default=8, help='input batch size for validing (default: 8)')
    parser.add_argument('--val_every', type=int, default=1, help='validation every {val_every}')
    parser.add_argument('--model', type=str, default='DeepLabV3Plus', help='model type (default: DeepLabV3Plus)')
    parser.add_argument('--encoder_name', type=str, default='senet154', help='model encoder type (default: SeNet154)')
    parser.add_argument('--encoder_weights', type=str, default='imagenet', help='model pretrain weight type (default: imagenet)')
    parser.add_argument('--in_channels', type=int, default=3, help='number of channels (default: 3)')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer type (default: Adam)')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')
    parser.add_argument('--lr_decay_step', type=int, default=5, help='learning rate scheduler deacy step (default: 5)')
    parser.add_argument('--name', type=str, default='Baseline Code', help='model save at')
    parser.add_argument('--save_limit', type=int, default=10, help='maximum limitation to save')
    parser.add_argument('--image_resize', type=int, default=224, help='resize image to train & val & test')
    parser.add_argument('--center_crop_size', type=int, default=0, help='center crop size, (default : 0)')
    parser.add_argument('--cutmix', type=bool, default=False, help='cutmix mode set, (default : False)')
    # parser.add_argument('--name', default='Baseline Code', help='model save at')

    # Container environment
    args = parser.parse_args()
    print(args)

    train(args)