import torch
import numpy as np
import os
import random
import argparse 
import timeit
from net.ERVSR import SRNet
from config import Config
from torch.utils.data import DataLoader
from utils.utils import *
from utils.datasets import *
from torchvision.utils import save_image
import warnings
warnings.filterwarnings("ignore")

seed = 216
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Parse arguments      
parser = argparse.ArgumentParser()
parser.add_argument('-device', '--device', type = str, default = 'cuda:0', help = 'device to run')
parser.add_argument('-model', '--model', type = str, help = 'path to saved model')
parser.add_argument('-data_offset', '--data_offset', type = str, help = 'root path of the dataset')
parser.add_argument('-save_img', '--save_img', type = bool, default=True, help = 'save image results')
parser.add_argument('-save_dir', '--save_dir', type = str, default='results/test_imgs',help = 'image logging directory')


param = parser.parse_args()

if param.save_img:
  os.makedirs(param.save_dir, exist_ok=True)

config = Config(param.data_offset, param.device)
config.spynet = None 

model = SRNet(config).to(param.device)
model.load_state_dict(torch.load(param.model))
print(toYellow('{} LOADED'.format(param.model)))

dataset_test = Test_datasets(config, is_valid=False)

if len(dataset_test)>0:
  print(toYellow('DATA LOADED'))
else: 
  raise RuntimeError("Dataset is not loaded")

test_loader = DataLoader(dataset_test, batch_size=1, shuffle=False)

print(toYellow('TEST START\n'))

psnr_vals = []

with torch.no_grad():
  for vid_i, inputs in enumerate(test_loader):
    # Prepare data
    LR_UW_total_frames = refine_image_pt(inputs['LR_UW'].to(param.device, non_blocking=True), 1)
    LR_REF_W_total_frames = refine_image_pt(inputs['LR_REF_W'], 1)
    LR_REF_W_frame = LR_REF_W_total_frames[:, LR_REF_W_total_frames.size(1)//2, :, :, :].to(param.device, non_blocking=True)
    HR_UW_total_frames = refine_image_pt(inputs['HR_UW'].to(param.device, non_blocking=True),1)
    _, total_frame_num, _, _, _ = LR_UW_total_frames.size()

    vid_vals = []
    start_t = timeit.default_timer()

    # Iterate frame
    for i in range(total_frame_num-(config.frame_num - 1)):
      LR_UW_frames = LR_UW_total_frames[:, i:i+config.frame_num]
      HR_UW_frames = HR_UW_total_frames[:, i:i+config.frame_num] 
      outs = model(LR_UW_frames, LR_REF_W_frame)
      
      psnr_val = psnr(outs.detach().cpu().numpy(), HR_UW_frames.detach().cpu().numpy())
      psnr_vals.append(psnr_val)
      vid_vals.append(psnr_val)
      
      if param.save_img:
        save_image(LR_UW_frames[0, outs.size(1)//2], os.path.join(param.save_dir, "{}_{}_LR.png".format(vid_i+1, i+1)))
        save_image(outs[0, outs.size(1)//2], os.path.join(param.save_dir, "{}_{}_SR.png".format(vid_i+1, i+1)))
        save_image(HR_UW_frames[0, outs.size(1)//2], os.path.join(param.save_dir, "{}_{}_HR.png".format(vid_i+1, i+1)))

    print_logs("TEST", vid_iter=vid_i, total_vids=len(test_loader), time=timeit.default_timer()-start_t, loss=np.mean(np.array(vid_vals)))

  print_psnr("TEST", mean_loss=np.mean(np.array(psnr_vals)))