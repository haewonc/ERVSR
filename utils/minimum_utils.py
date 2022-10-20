import numpy as np
import cv2
from PIL import Image
import math
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def psnr(img1, img2, PIXEL_MAX = 1.0):
    mse_ = np.mean( (img1 - img2) ** 2 )
    return 10 * math.log10(PIXEL_MAX / mse_)

def read_frame(path, norm_val = None, rotate_val = None, flip_val = None, gauss = None, gamma=0, sat_factor=None):
  if norm_val == (2**16-1):
      frame = cv2.imread(path, -1)
      frame = frame / norm_val
      frame = frame[...,::-1]
  else:
      frame = Image.open(path)

  frame = np.array(frame) / 255.

  if rotate_val is not None:
      frame = cv2.rotate(frame, rotate_val)
  if flip_val is not None:
      frame = cv2.flip(frame, flip_val)
  if gauss is not None:
      row,col,ch = frame.shape
      mean = 0
      gauss = np.random.normal(mean,1e-4,(row,col,ch))
      gauss = gauss.reshape(row,col,ch)

  frame = np.clip(frame, 0, 1.0)
  return frame

def refine_image_pt(image, val = 16):
    size = image.size()
    if len(size) == 5:
        h = size[3]
        w = size[4]
        return image[:, :, :, :h - h % val, :w - w % val]

    elif len(size) == 4:
        h = size[2]
        w = size[3]
        return image[:, :, :h - h % val, :w - w % val]

def get_tensor(paths = ['images/HR/*', 'images/LR/*', 'images/REF/*']):
    stacked = []
    for path in paths:
        imgs = []
        for img in glob.glob(path):
            t = torch.tensor(read_frame(img))
            refine_image_pt(t, 1)
            imgs.append(t)
        stacked.append(torch.stack(imgs).permute(0, 3, 2, 1))
    return stacked

def visualize_viewpoint_by_index(i, lr, ref):
    fig = plt.figure(dpi=300)
    gs=GridSpec(1,2)
    ax1 = fig.add_subplot(gs[0,0])
    ax1.imshow(lr[i].permute(2, 1, 0),aspect='auto')
    ax1.set_title('LR video frame')
    ax2 = fig.add_subplot(gs[0,1])
    ax2.imshow(ref[i].permute(2, 1, 0),aspect='auto')
    ax2.set_title('Reference image')
    for a in [ax1, ax2]:
        a.get_xaxis().set_visible(False)
        a.get_yaxis().set_visible(False)

def visualize_sr_by_index(i, lr_np, out_np, hr_np):
    fig = plt.figure(dpi=300)
    gs=GridSpec(4,9)
    ax1 = fig.add_subplot(gs[3, 0])
    ax1.imshow(lr_np[i].transpose(2, 1, 0),aspect='auto')
    ax1.set_title('LR video frame')
    ax2 = fig.add_subplot(gs[:, 1:5])
    ax2.imshow(out_np[i].transpose(2, 1, 0),aspect='auto')
    ax2.set_title('Super-resolved result')
    ax3 = fig.add_subplot(gs[:, 5:])
    ax3.imshow(hr_np[i].transpose(2, 1, 0),aspect='auto')
    ax3.set_title('Ground-truth image')
    for a in [ax1, ax2, ax3]:
        a.get_xaxis().set_visible(False)
        a.get_yaxis().set_visible(False)

def visualize_zoom_by_index(i, out_np, hr_np, zoom):
    fig = plt.figure(dpi=300)
    gs=GridSpec(1, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(out_np[i].transpose(2, 1, 0)[zoom[0]:zoom[1], zoom[2]:zoom[3]])
    ax1.set_title('Super-resolved result')
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(hr_np[i].transpose(2, 1, 0)[zoom[0]:zoom[1], zoom[2]:zoom[3]])
    ax2.set_title('Ground-truth image')
    for a in [ax1, ax2]:
        a.get_xaxis().set_visible(False)
        a.get_yaxis().set_visible(False)