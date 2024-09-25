import torch 
import numpy as np
from model.resnet_3d_2 import generate_model
from model.sfcn_rep import SFCN_rep
from torch.nn.functional import interpolate, pad
import pickle
import matplotlib.pyplot as plt 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
orig_shape = (256, 256, 64) # original shape of the image / mask 

def visualize_CAM(cam, image, save_path):
    # save CAM (layer 15, 20, 25, 30, 35, 40) into images
    resized_cam = cam
    fig, axs = plt.subplots(4, 3)
    pos_0_0 = axs[0, 0].imshow(resized_cam[...,15]*255, cmap='gray')
    fig.colorbar(pos_0_0, ax=axs[0, 0])
    axs[0, 0].set_title('layer 10')
    pos_0_1  = axs[0, 1].imshow(resized_cam[...,20]*255, cmap='gray')
    fig.colorbar(pos_0_1, ax=axs[0, 1])
    axs[0, 1].set_title('layer 20')
    pos_0_2 = axs[0, 2].imshow(resized_cam[...,25]*255, cmap='gray')
    fig.colorbar(pos_0_2, ax=axs[0, 2])
    axs[0, 2].set_title('layer 30')
    pos_1_0 = axs[1, 0].imshow(resized_cam[...,30]*255, cmap='gray')
    fig.colorbar(pos_1_0, ax=axs[1, 0])
    axs[1, 0].set_title('layer 40')
    pos_1_1 = axs[1, 1].imshow(resized_cam[...,35]*255, cmap='gray')
    fig.colorbar(pos_1_1, ax=axs[1, 1])
    axs[1, 1].set_title('layer 50')
    pos_1_2 = axs[1, 2].imshow(resized_cam[...,40]*255, cmap='gray')
    fig.colorbar(pos_1_2, ax=axs[1, 2])
    axs[1, 2].set_title('layer 60')
    pos_2_0 = axs[2, 0].imshow(image[...,15]*255, cmap='gray')
    # axs[2, 0].colorbar()
    axs[2, 0].set_title('image layer 10')
    pos_2_1 = axs[2, 1].imshow(image[...,20]*255, cmap='gray')
    # axs[2, 1].colorbar()
    axs[2, 1].set_title('image layer 20')
    pos_2_2 = axs[2, 2].imshow(image[...,25]*255, cmap='gray')
    # axs[2, 2].colorbar()
    axs[2, 2].set_title('image layer 30')
    pos_3_0 = axs[3, 0].imshow(image[...,30]*255, cmap='gray')
    # axs[3, 0].colorbar()
    axs[3, 0].set_title('image layer 40')
    axs[3, 1].imshow(image[...,35]*255, cmap='gray')
    # axs[3, 1].colorbar()
    pos_3_1 = axs[3, 1].set_title('image layer 50')
    axs[3, 2].imshow(image[...,40]*255, cmap='gray')
    # axs[3, 2].colorbar()
    pos_3_2 = axs[3, 2].set_title('image layer 60')
    # plt.colorbar()
    plt.savefig(save_path)
    return 

def get_model(exp_name='sfcn_rep1', fold=0):
    
    if 'sfcn_rep1' in exp_name:
        model = SFCN_rep(mode=1)
    elif 'sfcn_rep2' in exp_name:
        model = SFCN_rep(mode=2)
    else:
        pass
    
    model.load_state_dict(torch.load('/mnt/isilon/CSC4/HelenZhouLab/HZLHD1/Data4/Members/yileiwu/arwmc_reg/result/model/{}/fold{}/best_model.pth'.format(exp_name, fold), map_location=device))
    model = model.to(device)

    return model

def get_dice(image, target):
    return  2*sum(image.flatten()*target.flatten())/(sum(image.flatten()) + sum(target.flatten()))

def eval_dice(model:torch.nn.Module, intensity_percentile=99.4, cam_percentile=96.5):

    test_brain_mask = np.load("/mnt/isilon/CSC4/HelenZhouLab/HZLHD1/Data4/Members/yileiwu/arwmc_reg/preprocessed_data/test_brain_mask.npy", mmap_mode='r+')
    test_flair = np.load("/mnt/isilon/CSC4/HelenZhouLab/HZLHD1/Data4/Members/yileiwu/arwmc_reg/preprocessed_data/test.npy", mmap_mode='r+')
    test_mask = np.load("/mnt/isilon/CSC4/HelenZhouLab/HZLHD1/Data4/Members/yileiwu/arwmc_reg/preprocessed_data/test_mask.npy", mmap_mode='r+')
    test_arwmc = pickle.load(open("/mnt/isilon/CSC4/HelenZhouLab/HZLHD1/Data4/Members/yileiwu/arwmc_reg/preprocessed_data/test_target.p",'rb'))
    dice = []
    arwmc_pred = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_subjects = test_flair.shape[0]
    
    # a dict to store the activations
    activation = {}
    def getActivation(name):
        # the hook signature
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    h1 = model.layer4.register_forward_hook(getActivation('feature_map'))

    with torch.no_grad():
                    
        for i in range(num_subjects):
            temp_input = torch.from_numpy(np.expand_dims(test_flair[i], [0, 1]))
            temp_input = temp_input.to(device)
            temp_brain_mask = test_brain_mask[i, ...]
            temp_wmh_mask = test_mask[i, ...]

            temp_output0, cam0 = model(temp_input.float())

            # cam0 = torch.sum((cam0.detach()*model.fc.weight.unsqueeze(dim=2).unsqueeze(dim=2).unsqueeze(dim=2)), dim=1).unsqueeze(dim=0).cpu()
            cam0 = torch.nn.functional.interpolate(cam0.cpu(), size=test_flair[i,...].shape, mode='trilinear').numpy()[0, 0, ...] 
            #  plot
            # visualize_CAM(cam=cam0, image=test_flair[i], save_path='/mnt/isilon/CSC4/HelenZhouLab/HZLHD1/Data4/Members/yileiwu/arwmc_reg/raw_cam_vis/{}.png'.format(i))

            cam0 = np.where(cam0 > np.percentile(cam0.flatten(), cam_percentile), 1, 0)
    
            wmh_seg_pred = np.zeros(temp_brain_mask.shape)
            condition_cam0 = (cam0 > 0.5) * (temp_brain_mask > 1) * (test_flair[i,...] > np.percentile(test_flair[i,...].flatten(), intensity_percentile))
            wmh_seg_pred_cam0 = np.where(condition_cam0==1, 1, 0)

            dice.append(get_dice(wmh_seg_pred_cam0, temp_wmh_mask))
            # print(temp_output0.detach().cpu().numpy()[0][0])

            arwmc_pred.append(round(temp_output0.detach().cpu().view(-1).numpy()[0]))

    return np.mean(dice), np.std(dice), np.absolute(np.array(arwmc_pred) - np.array(test_arwmc)).mean(), np.absolute(np.array(arwmc_pred) - np.array(test_arwmc)).std()


# print(eval_dice(model))

# temp_input = torch.rand((1,1,256,256,64))
# temp_output = model(temp_input)
# print(activation['bn2.bias'].shape)

# for name, param in model.named_parameters():
#     print(name)
    # if name == 'fc.weight':
    #     print(param)
    #     print(param.size())
# print(model.fc.weight.size())

# intensity_percentile_pool = [98.8, 99.0, 99.2, 99.4, 99.6]
# cam_percentile_pool = [95.5, 96.0, 96.5, 97.0]
intensity_percentile_pool = [99.4]
cam_percentile_pool = [95.5]

for intensity_percentile in intensity_percentile_pool:
    for cam_percentile in cam_percentile_pool:
        print(intensity_percentile, cam_percentile, eval_dice(get_model(exp_name='sfcn_rep1', fold=0), intensity_percentile=intensity_percentile, cam_percentile=cam_percentile))
        print(intensity_percentile, cam_percentile, eval_dice(get_model(exp_name='sfcn_rep1', fold=1), intensity_percentile=intensity_percentile, cam_percentile=cam_percentile))
        print(intensity_percentile, cam_percentile, eval_dice(get_model(exp_name='sfcn_rep1', fold=2), intensity_percentile=intensity_percentile, cam_percentile=cam_percentile))
        print(intensity_percentile, cam_percentile, eval_dice(get_model(exp_name='sfcn_rep1', fold=3), intensity_percentile=intensity_percentile, cam_percentile=cam_percentile))
        print(intensity_percentile, cam_percentile, eval_dice(get_model(exp_name='sfcn_rep1', fold=4), intensity_percentile=intensity_percentile, cam_percentile=cam_percentile))

        print(intensity_percentile, cam_percentile, eval_dice(get_model(exp_name='sfcn_rep2', fold=0), intensity_percentile=intensity_percentile, cam_percentile=cam_percentile))
        print(intensity_percentile, cam_percentile, eval_dice(get_model(exp_name='sfcn_rep2', fold=1), intensity_percentile=intensity_percentile, cam_percentile=cam_percentile))
        print(intensity_percentile, cam_percentile, eval_dice(get_model(exp_name='sfcn_rep2', fold=2), intensity_percentile=intensity_percentile, cam_percentile=cam_percentile))
        print(intensity_percentile, cam_percentile, eval_dice(get_model(exp_name='sfcn_rep2', fold=3), intensity_percentile=intensity_percentile, cam_percentile=cam_percentile))
        print(intensity_percentile, cam_percentile, eval_dice(get_model(exp_name='sfcn_rep2', fold=4), intensity_percentile=intensity_percentile, cam_percentile=cam_percentile))

