# generate a report for the given study, I want:
# for each model from one split:
# 1. scatter plots (both validation and test) 
# 2. r2 score (correlation)
# 3. MSE score

from mypath import experiment_path
import os,re,datetime, torch, pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from medcam import medcam
import nibabel as nib
import glob
import torch.nn.functional as F 
import skimage.transform as skTrans

def get_model(model_type:str, task='reg'):

    if model_type == 'SFCN':
        from model.sfcn import SFCN
        model = SFCN(task=task)
    if model_type == 'SFCN4':
        from model.sfcn4 import SFCN_4
        model = SFCN_4(task=task)
    if model_type == 'SFCN3':
        from model.sfcn3 import SFCN_3
        model = SFCN_3(task=task)

    elif model_type == '3DResNet':
        from model.resnet import resnet50
        model = resnet50()
    elif model_type == 'ViT':
        if task == 'cls':
            from monai.networks.nets import ViT
            model = ViT(in_channels=1, img_size=(256, 256, 64), patch_size=16, classification=True, num_classes=30)
        else:
            from model.ViT_reg import ViT_reg
            model = ViT_reg(in_channels=1, img_size=(256, 256, 64), patch_size=16, classification=False)
    elif model_type == 'UNet':
        if task == 'reg':
            from model.unet import UNet_reg
            model = UNet_reg()

    return model

def get_preds(model, data, device, task='reg'):
    # return pred, given model (with weight) and data
    preds = []
    num_subjects = data.shape[0]
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        for i in range(num_subjects):
            temp_input = torch.from_numpy(np.expand_dims(data[i], [0, 1]))
            temp_input = temp_input.to(device)
            
            if task == 'reg':
                temp_output = model(temp_input.float()).view(-1).detach().cpu().numpy()[0]
                preds.append(temp_output)
            else:
                temp_output = model(temp_input.float()).view(-1).detach().cpu().numpy()
                centers = np.arange(0.5, 30.5)
                preds.append(sum(temp_output*centers))
    
    return np.array(preds)

def evaluate_study_regression(study_name:str):
    # given study name, 

    # select device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    # find the model type from config.txt 
    if os.path.exists(experiment_path + '/{}/config.txt'.format(study_name)):
        with open(experiment_path + '/{}/config.txt'.format(study_name), 'r') as f:
            model_type = re.search(r'model=\'\w+\'', f.read()).group(0)[7:-1]
    else:
        raise("config file not found error")

    with open(experiment_path + '/{}/report.txt'.format(study_name), 'a') as f:
        f.write('log time: {}\n'.format(datetime.datetime.now()))
        f.write('exp name: {}\n'.format(study_name))
        f.write('model name: {}\n'.format(model_type))

        for (dirpath, dirnames, filenames) in os.walk(experiment_path + '/{}'.format(study_name)):
            for split in dirnames:
                print(os.path.join(dirpath, split))
                f.write('\n{}\n'.format(split))

                if os.path.exists(experiment_path + '/{}/{}/best_model.pth'.format(study_name, split)):
                    model = get_model(model_type)
                    model.load_state_dict(torch.load(experiment_path + '/{}/{}/best_model.pth'.format(study_name, split), map_location=device))
                    model.eval()

                    # load validation dataset
                    flair_val = np.load('/mnt/isilon/CSC4/HelenZhouLab/HZLHD1/Data4/Members/yileiwu/arwmc_reg/preprocessed_data/{}_val.npy'.format(split))
                    mask_val = pickle.load(open('/mnt/isilon/CSC4/HelenZhouLab/HZLHD1/Data4/Members/yileiwu/arwmc_reg/preprocessed_data/{}_val_target.p'.format(split), 'rb')) 

                    # load test dataset
                    flair_test = np.load("/mnt/isilon/CSC4/HelenZhouLab/HZLHD1/Data4/Members/yileiwu/arwmc_reg/preprocessed_data/test.npy")
                    mask_test = pickle.load(open('/mnt/isilon/CSC4/HelenZhouLab/HZLHD1/Data4/Members/yileiwu/arwmc_reg/preprocessed_data/test_target.p', 'rb'))
                    # get prediction
                    preds_val = get_preds(model, flair_val, device)
                    preds_test = get_preds(model, flair_test, device)                    
                    # calcuate MSE and r2
                    mse_score_val = mean_squared_error(mask_val, preds_val)
                    mse_score_test = mean_squared_error(mask_test, preds_test)
                    r2_score_val = r2_score(mask_val, preds_val)
                    r2_score_test = r2_score(mask_test, preds_test)
                    f.write("Mean square Error for Validation Set is : {}\n".format(mse_score_val))
                    f.write("Mean square Error for Test Set is : {}\n".format(mse_score_test))
                    f.write("R2 score for Validation Set is : {}\n".format(r2_score_val))
                    f.write("R2 score for the test Set is: {}\n".format(r2_score_test))
                    f.write("GT for Val: {}\n".format(mask_val))
                    f.write("Preds for Val: {}\n".format(preds_val))
                    f.write("GT for Test: {}\n".format(mask_test))
                    f.write("Preds for Test: {}\n".format(preds_test))
                    # plot figure
                    plt.scatter(mask_val.astype(float), preds_val, marker="o", label="mse={:.3f}, r2={:.3f}".format(mse_score_val, r2_score_val))
                    plt.scatter(mask_test.astype(float), preds_test, marker="o", label="mse={:.3f}, r2={:.3f}".format(mse_score_test, r2_score_test))
                    plt.legend()
                    plt.savefig(experiment_path + '/{}/{}.png'.format(study_name, split))
                    plt.clf()

                else:
                    f.write("model not found !!! \n")

    pass 



# WIELD ISSUE: CANNOT WORK WITH GPU
def generate_CAM(study_name:str, fold:str):
    # generate CAM for all test subjects given study
    
    # select device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load test dataset
    flair_test = np.load("/mnt/isilon/CSC4/HelenZhouLab/HZLHD1/Data4/Members/yileiwu/arwmc_reg/preprocessed_data/test.npy")
    mask_test = pickle.load(open('/mnt/isilon/CSC4/HelenZhouLab/HZLHD1/Data4/Members/yileiwu/arwmc_reg/preprocessed_data/test_target.p', 'rb'))

    # find the model type from config.txt 
    if os.path.exists(experiment_path + '/{}/config.txt'.format(study_name)):
        with open(experiment_path + '/{}/config.txt'.format(study_name), 'r') as f:
            model_type = re.search(r'model=\'\w+\'', f.read()).group(0)[7:-1]
    else:
        raise("config file not found error")

    layer='feature_extractor'
    model_weight_path = experiment_path + '/{}/fold{}/best_model.pth'.format(study_name, fold)
    if os.path.exists(model_weight_path):
        model = get_model(model_type)
        model.load_state_dict(torch.load(model_weight_path, map_location=device))
        model = medcam.inject(model, output_dir=experiment_path+'/{}/fold{}/cam'.format(study_name, fold),layer=layer,  save_maps=True)
    model.eval()
    model = model.to(device)

    # get prediction
    get_preds(model, flair_test, device, task='reg')

    if not os.path.exists(experiment_path+'/{}/fold{}/cam/resized_cam'.format(study_name, fold)):
        os.mkdir(experiment_path+'/{}/fold{}/cam/resized_cam'.format(study_name, fold))
    
    if not os.path.exists(experiment_path+'/{}/fold{}/cam/flair_test'.format(study_name, fold)):
        os.mkdir(experiment_path+'/{}/fold{}/cam/flair_test'.format(study_name, fold))

    for each in glob.glob(experiment_path + '/{}/fold{}/cam/{}/*'.format(study_name, fold, layer)):
        temp_array = nib.load(each).get_fdata()
        resized_array = skTrans.resize(np.transpose(temp_array, (2, 0, 1)), (256, 256, 64))

        print(resized_array.shape)
        temp_image = nib.Nifti1Image(resized_array, affine=np.eye(4))
        temp_image.to_filename(experiment_path+'/{}/fold{}/cam/resized_cam/{}'.format(study_name, fold, each.split('/')[-1]))

    for i in range(flair_test.shape[0]):
        temp_array = flair_test[i]
        temp_image = nib.Nifti1Image(temp_array, affine=np.eye(4))
        temp_image.to_filename(experiment_path+'/{}/fold{}/cam/flair_test/{}.nii.gz'.format(study_name, fold, i))

# evaluate_study_regression('sfcn_mse_1e-3')
# evaluate_study_regression('sfcn4_mse_1e-3')
# evaluate_study_regression('sfcn3_mse_1e-3')

generate_CAM(study_name='sfcn3_mse_1e-3', fold='0')