import torch
import torch.nn as nn
import numpy as np
import os
import random
import argparse 
import timeit
from net.ERVSR import SRNet
from config import Config
import time
from torch.utils.data import DataLoader
from utils.utils import *
from utils.datasets import *
from torchvision.utils import save_image
import utils.lr_scheduler as lr_scheduler
import warnings
warnings.filterwarnings("ignore")

def iterate(loader, epoch, save_iter, is_train):
  if is_train:
    log_dir =  os.path.join(param.log_dir, "train_imgs")
    state = 'TRAIN'
  else:
    log_dir =  os.path.join(param.log_dir, "val_imgs")
    state = 'VALID'

  psnr_vals = []
  for vid_i, inputs in enumerate(loader):
    # Prepare data
    start_t = timeit.default_timer()
    LR_UW_total_frames = refine_image_pt(inputs['LR_UW'].to(param.device, non_blocking=True), 1)
    LR_REF_W_total_frames = refine_image_pt(inputs['LR_REF_W'], 1)
    LR_REF_W_frame = LR_REF_W_total_frames[:, LR_REF_W_total_frames.size(1)//2, :, :, :].to(param.device, non_blocking=True)
    HR_UW_total_frames = refine_image_pt(inputs['HR_UW'].to(param.device, non_blocking=True),1)
    _, total_frame_num, _, _, _ = LR_UW_total_frames.size()

    vid_losses = []
    # Iterate frame
    for i in range(total_frame_num-(config.frame_num - 1)):
      if is_train:
        optimizer.zero_grad()

      LR_UW_frames = LR_UW_total_frames[:, i:i+config.frame_num]
      HR_UW_frames = HR_UW_total_frames[:, i:i+config.frame_num] 
      outs = model(LR_UW_frames, LR_REF_W_frame)

      loss = l1_loss(outs, HR_UW_frames)

      if is_train:
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        scheduler.step()

      psnr_vals.append(psnr(outs.detach().cpu().numpy(), HR_UW_frames.detach().cpu().numpy()))
      vid_losses.append(psnr_vals[-1])

      # Save center frame
      if i == 0 and vid_i % save_iter == save_iter - 1:
        save_image(LR_UW_frames[0, outs.size(1)//2], os.path.join(log_dir, "{}_LR.png".format(vid_i+1)))
        save_image(outs[0, outs.size(1)//2], os.path.join(log_dir, "{}_SR.png".format(vid_i+1)))
        save_image(HR_UW_frames[0, outs.size(1)//2], os.path.join(log_dir, "{}_HR.png".format(vid_i+1)))
    print_logs(state, epoch+1, param.epoch, vid_i, len(loader), timeit.default_timer()-start_t, np.mean(np.array(vid_losses)))

  return np.mean(np.array(psnr_vals))

seed = 216
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Parse arguments      
parser = argparse.ArgumentParser()
parser.add_argument('-b', '--batch_size', type = int, default = 1, help = 'number of batch')
parser.add_argument('-epoch', '--epoch', type = int, default = 70, help = 'maximum number of epochs')
parser.add_argument('-device', '--device', type = str, default = 'cuda:0', help = 'device to run')
parser.add_argument('-log', '--log_dir', type=str, default='results/', help = 'root directory for logging')
parser.add_argument('-data_offset', '--data_offset', type = str, help = 'root path of the dataset')

param = parser.parse_args()

# Generate directories for logging
if not os.path.exists(param.log_dir):
  os.mkdir(param.log_dir)
os.makedirs(os.path.join(param.log_dir, "train_imgs"), exist_ok=True)
os.makedirs(os.path.join(param.log_dir, "val_imgs"), exist_ok=True)
os.makedirs(os.path.join(param.log_dir, "saved_models"), exist_ok=True)

config = Config(param.data_offset, param.device)
model = SRNet(config)
model = model.to(param.device)

model_time = time.strftime("%Y%m%d_%H%M")

optimizer = torch.optim.Adam([
  {'params': model.parameters(), 'lr': 2e-6, 'lr_init': 2e-6, 'betas':(0.9, 0.999), 'eps':1e-8}
], eps= 1e-8, lr=2e-6, betas=(0.9, 0.999))

learning_period = [0, param.epoch * (3456 // param.batch_size)]
scheduler = lr_scheduler.CosineAnnealingLR_Restart(
  optimizer, learning_period, eta_min= 1e-7,
  restarts= np.cumsum(learning_period)[:-1].tolist(), weights= np.ones_like(np.cumsum(learning_period)[:-1].tolist()).tolist()
)
l1_loss = nn.L1Loss()

dataset_train = Train_datasets(config) 
dataset_val = Test_datasets(config, is_valid=True)

if len(dataset_train)>0 and len(dataset_val)>0:
  print(toYellow('DATA LOADED'))
else: 
  raise RuntimeError("Dataset is not loaded")

train_loader = DataLoader(dataset_train, batch_size=param.batch_size, shuffle=True, num_workers=4, pin_memory=True)
valid_loader = DataLoader(dataset_val, batch_size=1, shuffle=False)

print(toYellow('TRAINING START\n'))

for epoch in range(15, param.epoch):
  model.train()
  train_losses = []
  mean_loss = iterate(train_loader, epoch, config.train_img_save, True)
  print_psnr("TRAIN", epoch+1, param.epoch, mean_loss)

  if epoch % config.valid_iter == config.valid_iter-1:
    model.eval()
    with torch.no_grad():
      mean_loss = iterate(valid_loader, epoch, config.valid_img_save, False)
      print_psnr("VALID", epoch+1, param.epoch, mean_loss)
  
  if epoch % config.model_save_iter == config.model_save_iter-1:
    model_name = "model_{}_epoch_{}.pt".format(model_time, epoch+1)
    print(toRed('Model saved as {}'.format(model_name)))
    torch.save(model.state_dict(), os.path.join(os.path.join(param.log_dir, "saved_models", model_name)))

print(toYellow('\nTRAINING END\n'))
model_name = "model_{}_final.pt".format(model_time)
torch.save(model.state_dict(), os.path.join(os.path.join(param.log_dir, "saved_models", model_name)))
print(toYellow('\FINAL MODEL SAVED\n'))