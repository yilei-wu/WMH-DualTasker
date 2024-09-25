import argparse
parser = argparse.ArgumentParser(description = "Pytorch WMH segmentation Evauation")

parser.add_argument('--model', type=str, default='SFCN', choices=['SFCN', 'SFCN4', 'SFCN3', '3DResNet', 'ViT'])
parser.add_argument('--task', type=str, default='reg', choices=['reg', 'cls'])

parser.add_argument('--exp_name', type=str, required=True)
parser.add_argument('--fold', type=int, default=0, nargs='+', required=True)

parser.add_argument('--save_path', type=str, default=None)

args = parser.parse_args()

import numpy as np 
import pickle 
from mypath import experiment_path
import torch
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

for fold in args.fold:

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # flair_test = np.load('/mnt/isilon/CSC4/HelenZhouLab/HZLHD1/Data4/Members/yileiwu/arwmc_reg/preprocessed_data/fold{}_val.npy'.format(fold))
    # mask_test = pickle.load(open('/mnt/isilon/CSC4/HelenZhouLab/HZLHD1/Data4/Members/yileiwu/arwmc_reg/preprocessed_data/fold{}_val_target.p'.format(fold), 'rb')) 

    # for test dataset
    flair_test = np.load("/mnt/isilon/CSC4/HelenZhouLab/HZLHD1/Data4/Members/yileiwu/arwmc_reg/preprocessed_data/test.npy")
    mask_test = pickle.load(open('/mnt/isilon/CSC4/HelenZhouLab/HZLHD1/Data4/Members/yileiwu/arwmc_reg/preprocessed_data/test_target.p', 'rb'))

    if args.model == 'SFCN':
        from model.sfcn import SFCN
        model = SFCN(task=args.task)
    if args.model == 'SFCN4':
        from model.sfcn4 import SFCN_4
        model = SFCN_4(task=args.task)
    if args.model == 'SFCN3':
        from model.sfcn3 import SFCN_3
        model = SFCN_3(task=args.task)
    elif args.model == '3DResNet':
        from model.resnet import resnet50
        model = resnet50()
    elif args.model == 'ViT':
        from monai.networks.nets import ViT
        model = ViT(in_channels=1, img_size=(256, 256, 64), patch_size=16, classification=True, num_classes=30)

    model_weight_path = experiment_path + '/' + args.exp_name + '/fold{}/best_model.pth'.format(fold)
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model = model.to(device)
    preds = []

    num_subjects = flair_test.shape[0]
    for i in range(num_subjects):
        temp_input = torch.from_numpy(np.expand_dims(flair_test[i], [0, 1]))
        temp_input = temp_input.to(device)
        
        if args.task == 'reg':
            print(temp_input.shape)
            temp_output = model(temp_input.float()).view(-1).detach().cpu().numpy()[0]
            preds.append(temp_output)
        else:
            temp_output = model(temp_input.float()).view(-1).detach().cpu().numpy()
            centers = np.arange(0.5, 30.5)
            preds.append(sum(temp_output*centers))
    print(preds)

    plt.scatter(mask_test, preds)
    plt.savefig("/mnt/isilon/CSC4/HelenZhouLab/HZLHD1/Data4/Members/yileiwu/arwmc_reg/{}_fold{}.png".format(args.exp_name, fold))

    r2 = r2_score(mask_test, preds)
    print(r2)
