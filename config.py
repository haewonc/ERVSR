import os
import numpy as np

class Config:
    def __init__(self, data_offset, device):

        ## Network 
        self.num_feat = 64
        self.num_blocks = 30
        self.enc_num_blocks = 16
        self.mid_channels = 36
        self.data_offset = data_offset
        self.device = device
        self.spynet = 'net/SPyNet.pytorch'
        data = 'RealMCVSR'
        
        ## Training
        self.batch_size = 8
        self.patch_size = 64
        self.frame_itr_num = 9
        self.frame_num = 7
        self.window_size = 6

        ## Dataset
        self.is_use_T = False
        self.flag_HD_in = False 
        self.is_crop = True
        self.is_crop_valid = False
        self.scale = 4 
        if self.scale == 2:
            self.matching_ksize = 4 
        else:
            self.matching_ksize = 2 
        self.is_crop_valid = False
        if self.flag_HD_in:
            self.matching_ksize *= self.scale

        ## Data path 
        lr_path = 'LRx4'
        hr_ref_W_path = 'LRx2'
        hr_ref_T_path = 'LRx4'

        self.LR_data_path = os.path.join(self.data_offset, data, 'train', lr_path)
        self.HR_data_path = os.path.join(self.data_offset, data, 'train', 'HR')
        self.HR_ref_data_W_path = os.path.join(self.data_offset, data, 'train', hr_ref_W_path)
        self.HR_ref_data_T_path = os.path.join(self.data_offset, data, 'train', hr_ref_T_path)

        self.VAL_LR_data_path = os.path.join(self.data_offset, data, 'valid', lr_path)
        self.VAL_HR_data_path = os.path.join(self.data_offset, data, 'valid', 'HR')
        self.VAL_HR_ref_data_W_path = os.path.join(self.data_offset, data, 'valid', hr_ref_W_path)
        self.VAL_HR_ref_data_T_path = os.path.join(self.data_offset, data, 'valid', hr_ref_T_path)
        
        self.EVAL_test_set = 'test'
        self.EVAL_LR_data_path = os.path.join(self.data_offset, data, self.EVAL_test_set, lr_path)
        self.EVAL_HR_data_path = os.path.join(self.data_offset, data, self.EVAL_test_set, 'HR')
        self.EVAL_HR_ref_data_W_path = os.path.join(self.data_offset, data, self.EVAL_test_set, hr_ref_W_path)
        self.EVAL_HR_ref_data_T_path = os.path.join(self.data_offset, data, self.EVAL_test_set, hr_ref_T_path)
        self.EVAL_vid_name = None 

        self.UW_path = 'UW'
        self.W_path = 'W'
        self.T_path = 'T'

        ## Log
        self.valid_iter = 5
        self.model_save_iter = 5
        self.train_img_save = 30
        self.valid_img_save = 10

        ## Legacy 
        self.dist = False