import numpy as np 
import pickle 
import torch
from model.sfcn import SFCN
from model.sfcn4 import SFCN_4
from model.sfcn3 import SFCN_3
from medcam import medcam

flair_test = np.load("/mnt/isilon/CSC4/HelenZhouLab/HZLHD1/Data4/Members/yileiwu/arwmc_reg/preprocessed_data/test.npy")
mask_test = pickle.load(open('/mnt/isilon/CSC4/HelenZhouLab/HZLHD1/Data4/Members/yileiwu/arwmc_reg/preprocessed_data/test_target.p', 'rb'))

model_weight_path = "/mnt/isilon/CSC4/HelenZhouLab/HZLHD1/Data4/Members/yileiwu/arwmc_reg/result/sfcn3_mse_1e-3/fold0/best_model.pth"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = SFCN_3(task = 'reg')
model.load_state_dict(torch.load(model_weight_path, map_location=device))
model = medcam.inject(model, output_dir="/mnt/isilon/CSC4/HelenZhouLab/HZLHD1/Data4/Members/yileiwu/arwmc_reg/saliency_map_sfcn4", layer='feature_extractor', save_maps=True)
model.eval()

temp_input = torch.from_numpy(np.expand_dims(flair_test[0], [0, 1]))
temp_input = temp_input.to(device)

temp_output = model(temp_input.float()).view(-1).detach().cpu().numpy()[0]

print(temp_output)