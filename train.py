import argparse
parser = argparse.ArgumentParser(description="ARWMC visual rating regression")

#model related 
parser.add_argument('--model', type=str, default='SFCN', choices=['SFCN', 'SFCN4', 'SFCN3', '3DResNet', 'ViT', 'UNet', 'model1', 'model2', 'resnet50', 'resnet34', 'resnet18', 'resnet10', 'resnet50_2', 'resnet34_2', 'resnet18_2', 'resnet10_2', 'sfcn_rep1', 'sfcn_rep2'])

#training related
parser.add_argument('--data_augmentation', dest='data_augmentation', default="DA", type=str, choices=['noDA', 'DA'])
parser.add_argument('--fold', type=int, nargs='+', default=[0, 1, 2, 3, 4])
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--optimizer', type=str, default='sgd', choices=['adam', 'sgd'])
parser.add_argument('--loss', type=str, default='MSE', choices=['MSE', 'KLD', 'weighted_MSE', 'weighted_Focal'])
parser.add_argument('--task', type=str, default='cls', choices=['cls', 'reg'])
parser.add_argument('--epoch',  type=int, default=400)
parser.add_argument('--random_seed', type=int, default=42)
parser.add_argument('--experiment_name', type=str, default='sample_name')
args = parser.parse_args()

use_t1 = False
if args.data_augmentation == 'DA':
    args.epoch = 100
#A bunch of dependency import
import numpy as np
import torch, os, time, copy, csv, random, sys, datetime
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler
import gc
import pickle

from mypath import experiment_path, macc
from dataset.dataset import my_dataset, my_dataset_V2
from training.loss import my_MSELoss, my_KLDLoss, my_weighted_MSELoss
from training.trainer import train_model
from scipy.ndimage import gaussian_filter1d
# enforce random seed
seed = args.random_seed
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def get_lds_kernel_window(kernel, ks, sigma):
    assert kernel in ['gaussian', 'triang', 'laplace']
    half_ks = (ks - 1) // 2
    if kernel == 'gaussian':
        base_kernel = [0.] * half_ks + [1.] + [0.] * half_ks
        kernel_window = gaussian_filter1d(base_kernel, sigma=sigma) / max(gaussian_filter1d(base_kernel, sigma=sigma))
    elif kernel == 'triang':
        kernel_window = triang(ks)
    else:
        laplace = lambda x: np.exp(-abs(x) / sigma) / (2. * sigma)
        kernel_window = list(map(laplace, np.arange(-half_ks, half_ks + 1))) / max(map(laplace, np.arange(-half_ks, half_ks + 1)))

    return kernel_window

# determine the result path 
experiment_log = os.path.join(experiment_path, args.experiment_name)
if not os.path.exists(experiment_log):
    os.makedirs(experiment_log)
with open(experiment_log + '/config.txt', 'w') as config_file:
    config_file.write(str(args))

for fold in args.fold:
    fold_log_path = experiment_log + '/fold{}'.format(fold)
    if not os.path.exists(fold_log_path):
      os.makedirs(fold_log_path)
    
    if use_t1:
        flair_train = np.load(macc + '/fold{}_train.npy'.format(fold), mmap_mode='r')
        t1_train = np.load(macc + '/fold{}_train_t1.npy'.format(fold), mmap_mode='r')
        mask_train = pickle.load(open(macc + '/fold{}_train_target.p'.format(fold), 'rb'))
        flair_test = np.load(macc + '/fold{}_val.npy'.format(fold), mmap_mode='r')
        t1_test = np.load(macc + '/fold{}_val_t1.npy'.format(fold), mmap_mode='r')
        mask_test = pickle.load(open(macc + '/fold{}_val_target.p'.format(fold), 'rb'))

    else:
        flair_train = np.load(macc + '/fold{}_train.npy'.format(fold), mmap_mode='r')
        t1_train = None
        mask_train = pickle.load(open(macc + '/fold{}_train_target.p'.format(fold), 'rb'))
        flair_test = np.load(macc + '/fold{}_val.npy'.format(fold), mmap_mode='r')
        t1_test = None
        mask_test = pickle.load(open(macc + '/fold{}_val_target.p'.format(fold), 'rb'))

    # prepare the dataloader
    if args.data_augmentation == 'noDA':
        train_dataset = my_dataset_V2(flair_train, mask_train, t1=t1_train)
        test_dataset = my_dataset(flair_test, mask_test, t1=t1_test)

        train_datasetloader = DataLoader(train_dataset, batch_size=args.batch_size,num_workers=0, pin_memory=True,shuffle=True)
        test_datasetloader = DataLoader(test_dataset, batch_size=1,num_workers=0, pin_memory=True, shuffle=True)

        dataloaders = {'Train': train_datasetloader, 'Test': test_datasetloader}

    if args.data_augmentation == 'DA':
        from dataset.dataloader import DataLoader4BG
        from dataset.augmentation import get_default_augmentation
        train_dataset = DataLoader4BG(image=flair_train, ratings=mask_train, batch_size=args.batch_size, num_threads_in_mt=12)
        test_dataset = my_dataset(flair_test, mask_test)

        train_datasetloader = get_default_augmentation(train_dataset)
        test_datasetloader = DataLoader(test_dataset, batch_size=1,num_workers=8, pin_memory=True, shuffle=True)

        dataloaders = {'Train': train_datasetloader, 'Test': test_datasetloader}

    # select the correspond model to import 
    if args.model == 'SFCN':
        from model.sfcn import SFCN
        model = SFCN(task=args.task)
    if args.model == 'SFCN4':
        from model.sfcn4 import SFCN_4
        model = SFCN_4(task=args.task)
    if args.model == 'SFCN3':
        from model.sfcn3 import SFCN_3
        model = SFCN_3(task=args.task)
    if args.model == 'model1':
        from model.pool_twice_net import pool_twice_net
        model = pool_twice_net()
    if args.model == 'model2':
        from model.pool_twice_net import pool_twice_net_res
        model = pool_twice_net_res()
    elif args.model == '3DResNet':
        from model.resnet import resnet50
        model = resnet50()
    elif args.model == 'ViT':
        if args.task == 'cls':
            from monai.networks.nets import ViT
            model = ViT(in_channels=1, img_size=(256, 256, 64), patch_size=16, classification=True, num_classes=30)
        else:
            from model.ViT_reg import ViT_reg
            model = ViT_reg(in_channels=1, img_size=(256, 256, 64), patch_size=16, classification=False)
    elif args.model == 'UNet':
        if args.task == 'reg':
            from model.unet import UNet_reg
            model = UNet_reg()
    elif args.model == 'resnet50':
        from model.resnet_3d import generate_model
        model = generate_model(50)
    elif args.model == 'resnet34':
        from model.resnet_3d import generate_model
        model = generate_model(34)
    elif args.model == 'resnet18':
        from model.resnet_3d import generate_model
        model = generate_model(18)
    elif args.model == 'resnet10':
        from model.resnet_3d import generate_model
        model = generate_model(10)
    elif args.model == 'resnet50_2':
        from model.resnet_3d_2 import generate_model
        model = generate_model(50)
    elif args.model == 'resnet34_2':
        from model.resnet_3d_2 import generate_model
        model = generate_model(34)
    elif args.model == 'resnet18_2':
        from model.resnet_3d_2 import generate_model
        model = generate_model(18)
    elif args.model == 'resnet10_2':
        from model.resnet_3d_2 import generate_model
        model = generate_model(10)
    elif args.model == 'sfcn_rep1':
        from model.sfcn_rep import SFCN_rep
        model = SFCN_rep(mode=1)
    elif args.model == 'sfcn_rep2':
        from model.sfcn_rep import SFCN_rep
        model = SFCN_rep(mode=2)
    model.train()
    
    weights = None
    # select loss fucntion
    if args.loss == 'MSE':
        criterion = my_MSELoss()
        # criterion = torch.nn.MSELoss()
    if args.loss == 'KLD':
        criterion = my_KLDLoss()
    if args.loss == 'weighted_MSE' or args.loss == 'weighted_Focal':
        from collections import Counter
        from scipy.ndimage import convolve1d
        
        bin_index_per_label = np.array(mask_train, dtype=np.uint8)
        Nb = max(bin_index_per_label) + 1
        num_samples_of_bins = dict(Counter(bin_index_per_label))
        emp_label_dist = [num_samples_of_bins.get(i, 0) for i in range(Nb)]
        # print(Nb)
        # print(emp_label_dist)
        lds_kernel_window = get_lds_kernel_window(kernel='gaussian', ks=5, sigma=2)
        eff_label_dist = convolve1d(np.array(emp_label_dist), weights=lds_kernel_window, mode='constant')
        print(eff_label_dist)
        eff_num_per_label = [eff_label_dist[bin_idx] for bin_idx in bin_index_per_label]
        weights = np.array([np.float32(1 / x) for x in eff_num_per_label])
        
        if args.loss == 'weighted_MSE':
            criterion = my_weighted_MSELoss()
        if args.loss == 'weighted_Focal':
            criterion = my_weighted_FocalLoss()

    # select optimizer & scheduler
    if args.optimizer == 'adam':
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    elif args.model == 'sfcn_rep1':
        optimizer = torch.optim.SGD(model.parameters(), lr=0.00001, weight_decay=5e-4, momentum=0.99, nesterov=True)
    else:
        # nnunet optimizer
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0005, weight_decay=1e-3, momentum=0.99, nesterov=True)

    metrics = []
    best_model = train_model(model, criterion, dataloaders, optimizer, metrics=metrics, bpath=fold_log_path, args=args, weights=weights)

    # release cuda cache
    # explicitly free all memory
    del flair_train
    del flair_test
    del mask_train
    del mask_test
    del best_model
    del model
    del optimizer
    del criterion
    del test_datasetloader
    del train_datasetloader
    gc.collect()
    torch.cuda.empty_cache()