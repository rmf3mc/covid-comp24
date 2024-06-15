# train.py

import torch
from tqdm import tqdm
import gc
from sklearn.metrics import f1_score, roc_auc_score
import numpy as np
import copy
import os
from torch.cuda import amp

from config import CONFIG

# Time for measuring duration
import time

# Collections for defaultdict
from collections import defaultdict

def train_one_epoch(criterion,scaler, model, optimizer, scheduler, dataloader, device, epoch):
    model.train()

    dataset_size = 0
    running_loss = 0.0

    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    print(f'Epoch:{epoch}')
    
    for step, data in bar:
        ct_b, img_b, c, h, w = data['image'].size()
        
        data_img = data['image'].reshape(-1, c, h, w)
        data_label = data['label'].reshape(-1,1)
        
        images = data_img.to(device, dtype=torch.float)
        labels = data_label.to(device, dtype=torch.float)


        batch_size = images.size(0)

        with amp.autocast(enabled = True):


            outputs = model(images)
            loss = criterion(outputs, labels)
            
        scaler.scale(loss).backward()


        scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()


        if scheduler is not None:
            scheduler.step()

        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size

        bar.set_postfix(Epoch=epoch, Train_Loss=epoch_loss,  LR=optimizer.param_groups[0]['lr'])
    gc.collect()

    return epoch_loss
    
    
@torch.inference_mode()
def valid_one_epoch(optimizer, criterion, model, dataloader, device, epoch):
    model.eval()

    dataset_size = 0
    running_loss = 0.0

    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    true_y=[]
    pred_y=[]
    for step, data in bar:
        ct_b, img_b, c, h, w = data['image'].size()
        
        data_img = data['image'].reshape(-1, c, h, w)
        data_label = data['label'].reshape(-1,1)

        images = data_img.to(device, dtype=torch.float)
        labels = data_label.to(device, dtype=torch.float)

        batch_size = images.size(0)

        outputs = model(images)
        loss = criterion(outputs, labels)


        true_y.append(labels.cpu().numpy())
        pred_y.append(torch.sigmoid(outputs).cpu().numpy())

        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size

        bar.set_postfix(Epoch=epoch, Valid_Loss=epoch_loss, LR=optimizer.param_groups[0]['lr'])

    true_y=np.concatenate(true_y)
    pred_y=np.concatenate(pred_y)

    gc.collect()

    true_y=np.array(true_y).reshape(-1,1)
    true_y=np.array(true_y).reshape(-1,img_b)
    true_y=true_y.mean(axis=1)

    pred_y=np.array(pred_y).reshape(-1,1)
    pred_y = torch.nan_to_num(torch.from_numpy(pred_y)).numpy()
    pred_y=np.array(pred_y).reshape(-1,img_b)

    pred_y=pred_y.mean(axis=1)

    acc_f1=f1_score(np.array(true_y),np.round(pred_y),average='macro')
    auc_roc=roc_auc_score(np.array(true_y),np.array(pred_y))
    
    print("acc_f1(mean) : ",round(acc_f1,4),"  auc_roc(mean) : ",round(auc_roc,4))

    return epoch_loss,acc_f1,auc_roc
    
def run_training(train_loader,valid_loader,criterion,scaler,model, bin_save_path, job_name, optimizer, scheduler, device, num_epochs):


    if torch.cuda.is_available():
        print("[INFO] Using GPU: {}\n".format(torch.cuda.get_device_name()))

    start = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_epoch_loss = np.inf
    best_epoch_auc = 0
    best_epoch_f1 = 0
    history = defaultdict(list)
    for epoch in range(1, num_epochs + 1):
        gc.collect()
        
        train_epoch_loss = train_one_epoch(criterion,scaler, model, optimizer, scheduler, dataloader=train_loader, device=CONFIG['device'], epoch=epoch)
        
        val_epoch_loss,acc_f1,auc_roc= valid_one_epoch(optimizer, criterion, model, valid_loader, device=CONFIG['device'],epoch=epoch)


        history['Train Loss'].append(train_epoch_loss)
        history['Valid Loss'].append(val_epoch_loss)

        if auc_roc >= best_epoch_auc:
            print(f"Validation Auc Improved ({best_epoch_auc} ---> {auc_roc})")
            best_epoch_auc = auc_roc

            best_model_wts = copy.deepcopy(model.state_dict())
            PATH = f'{bin_save_path}/auc_roc_ECA/'+job_name
            os.makedirs(f'{bin_save_path}/auc_roc_ECA/', exist_ok=True)
            torch.save(model.state_dict(), PATH)
            # Save a model file from the current directory
            print(f"Model Saved")

        if acc_f1 >= best_epoch_f1:
            print(f"Validation f1 Improved ({best_epoch_f1} ---> {acc_f1})")
            best_epoch_f1 = acc_f1

            best_model_wts = copy.deepcopy(model.state_dict())
            PATH = f'{bin_save_path}/f1_ECA/'+job_name
            os.makedirs(f'{bin_save_path}/f1_ECA/', exist_ok=True)
            torch.save(model.state_dict(), PATH)
            # Save a model file from the current directory
            print(f"Model Saved")

    end = time.time()
    time_elapsed = end - start
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
    print("Best Loss: {:.4f}".format(best_epoch_loss))
    return model, history