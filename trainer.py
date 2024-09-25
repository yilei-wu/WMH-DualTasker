import numpy as np
import os
import torch
import sys
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
import time, copy, csv
from tqdm import tqdm
import torch.nn.functional as F
from training.utils import num2vect
sys.path.insert(0, '/mnt/isilon/CSC4/HelenZhouLab/HZLHD1/Data4/Members/yileiwu/WSSS/WMH/trial_1')
from utils import eval_dice, visualize_CAM

def train_model(model, criterion, dataloaders, optimizer, metrics, bpath, args, weights):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    kwargs = {'epoch':args.epoch, 'task':args.task, 'model_type':args.model, 'loss':args.loss}
    
    best_loss = 1e10
    best_acc = 0
    # Use gpu if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # Initialize the log file for training and testing loss and metrics
    fieldnames = ['epoch', 'Train_loss', 'Test_loss'] + [f'Train_{m}' for m in metrics] + [f'Test_{m}' for m in metrics]
    with open((bpath + '/train_log.csv'), 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    
    if not os.path.exists(bpath + '/cam_vis'):
        os.mkdir(bpath + '/cam_vis')
    
    try:

        for epoch in range(1, kwargs['epoch'] + 1):
            print('Epoch {}/{}'.format(epoch, kwargs['epoch']))
            print('-' * 10)
            
            # Each epoch has a training and validation phase
            # Initialize batch summary
            batchsummary = {a: [0] for a in fieldnames}

            for phase in ['Train', 'Test']:
                if phase == 'Train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluate mode

                loss_list = [] # accumulate the loss throughout one epoch

                # Iterate over data only train phase in the current dataloader
                for i, sample in tqdm(enumerate(dataloaders[phase])):
                    # print("hello")
                    inputs = sample['image'].to(device, dtype=torch.float)
                    ratings = sample['rating'].to(device, dtype=torch.float)
                    # print(ratings)
                    # print(inputs.shape)
                    # zero the parameter gradients
                    optimizer.zero_grad()

                    if epoch % 5 == 1: 
                        # every five epoches
                        gen_vis = True
                        if not os.path.exists(bpath + '/cam_vis/Epoch_{}'.format(epoch)):
                            os.mkdir(bpath + '/cam_vis/Epoch_{}'.format(epoch))

                    else:
                        gen_vis = False 

                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'Train'):

                        if kwargs['model_type']=='ViT':
                            # softmax_layer = torch.nn.Softmax(dim=1)
                            # out = softmax_layer(model(inputs)[0])
                            out, cam = model(inputs)
                            out = out.view(out.shape[0], -1) # batch_size * 30 (cls), batch_size * 1 (reg)
                            # print(out)
                        else:
                            # softmax_layer = torch.nn.Softmax(dim=1)
                            # out = softmax_layer(model(inputs))
                            out, cam = model(inputs)
                            out = out.view(out.shape[0], out.shape[1]) # batch_size * 30 (cls), batch_size * 1 (reg)
                            # print(out)
                        
                        if kwargs['task'] == 'cls':
                            if kwargs['loss'] == 'MSE':
                                centers = torch.range(0.5, 29.5).view(-1).to(device)
                                out = torch.sum((out*centers), dim=1)
                            else:
                                # use KLD loss, convert 
                                ratings = num2vect(ratings, 1).to(device, dtype=torch.float)

                        # elif kwargs['task'] == 'reg':
                            # take argmax
                            # cls_out = torch.argmax(out, dim=1)
                        # print(out)
                        out = out.to(device, dtype=torch.float)
                        ratings = ratings.to(device, dtype=torch.float)
                        
                        if criterion.name == 'weighted_MSELoss':
                            loss = criterion(out, ratings, weights[sample['idx'].view(-1)])
                        else:
                            loss = criterion(out, ratings)
                        
                        loss_list.append(loss.item())

                        # backward + optimize only if in training phase
                        if phase == 'Train':
                            loss.backward()
                            optimizer.step()
                            del loss

                        if phase == 'Test' and gen_vis:
                            cam = F.interpolate(cam, size=(inputs.shape[2], inputs.shape[3], inputs.shape[4]), mode='trilinear')
                            visualize_CAM(cam[0:1,...].detach().cpu(), inputs[0:1,...].detach().cpu(), bpath + '/cam_vis/Epoch_{}/{}_cam.png'.format(epoch, i))
            
                batchsummary['epoch'] = epoch
                epoch_loss = sum(loss_list) / len(loss_list) # average the loss within one epoch
                batchsummary[f'{phase}_loss'] = epoch_loss
            
            with open((bpath + '/train_log.csv'), 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow(batchsummary)
                if phase == 'Test' and batchsummary['Test_loss'] < best_loss:
                    best_loss = batchsummary['Test_loss']
                    best_model_wts = copy.deepcopy(model.state_dict())
                    
                    model_save_path = bpath + '/best_model.pth'
                    torch.save(best_model_wts, model_save_path)
                    print('update ! Epoch {} \n'.format(epoch))
                    with open((bpath + '/train_log.csv'), 'a', newline='') as f:
                        f.write("update wirh test loss ")
                        f.write('Segmentation on test set {} \n'.format(eval_dice(model)))
    except RuntimeError as e:
        print(e)
        print("runtime error")
        print("runtime error")
        print("runtime error")

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Lowest Loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model