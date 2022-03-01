#================================================================================
#================================================================================
from re import L
import numpy as np
import cv2
import os
from glob import glob
import nibabel as nib
import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from Utility import window_center_adjustment
import Config
#================================================================================
#================================================================================

#================================================================================
def dataset_slicer(root_path, output_root):
    '''
        Slice 3D images in given path to 2D images.
        We use this function to generate required data to train 2D UNet network.
    '''

    #create ouput path if not exists
    if os.path.exists(output_root) is not True:
        os.makedirs(output_root);

    #list all folders in path
    folder_in_path = glob(root_path + "\\*");

    #for each folder in path, open "anat" folder which contains MRI and its segmentation
    for f in folder_in_path:
        #change directory convection in case of UNIX base system
        f = f.replace("/","\\");

        #list nifti images
        path_to_images = os.path.sep.join([f, "anat"]);
        nifti_files = glob(path_to_images + "\\*.gz");

        for n in nifti_files:
            #if name of file contains "dseg", it means that it is a segmentation file
            if "dseg" in n:
                mri_segmentation = nib.load(n).get_fdata();
            else:
                mri_3d = nib.load(n).get_fdata();
        
        assert mri_segmentation.shape == mri_3d.shape;

        #slice each image and save them in appropriate folder in output root
        folder_name = f[f.rfind("\\")+1:];

        path_to_save = os.path.sep.join([output_root, folder_name]);

        #create folder name if not exists
        if os.path.exists(path_to_save) is False:
            os.makedirs(path_to_save);

        #counter for name of the files
        cnt = 1;
        max_val_volume = np.max(mri_3d);
        #loop over first view
        for i in range(mri_segmentation.shape[0]):
                seg_slice = mri_segmentation[i,:,:];
                mri_slice = window_center_adjustment(mri_3d[i,:,:], max_val_volume);
                d = [mri_slice, seg_slice];
                pickle.dump(d, open(os.path.sep.join([path_to_save, f"{cnt}.ml"]), "wb"));
                cnt += 1;
        
        #loop over second view
        for i in range(mri_segmentation.shape[1]):
                seg_slice = mri_segmentation[:,i,:];
                mri_slice = window_center_adjustment(mri_3d[:,i,:], max_val_volume);
                d = [mri_slice, seg_slice];
                pickle.dump(d, open(os.path.sep.join([path_to_save, f"{cnt}.ml"]), "wb"));
                cnt += 1;
        
        #loop over third view
        for i in range(mri_segmentation.shape[2]):
                seg_slice = mri_segmentation[:,:,i];
                mri_slice = window_center_adjustment(mri_3d[:,:,i], max_val_volume);
                d = [mri_slice, seg_slice];
                pickle.dump(d, open(os.path.sep.join([path_to_save, f"{cnt}.ml"]), "wb"));
                cnt += 1;
#================================================================================

#================================================================================
def load_entire_MRI(root_path):
    '''
        This function loads the entire 2D MRI in given directories and return mask and MRI images
    '''
    total_data = [];
    for p in root_path:
        p = str(p);
        data_in_dir = glob(p + "\\*.ml");
        total_data.append(data_in_dir);
    
    total_data = np.array(total_data);

    total_data = total_data.reshape(total_data.shape[0]*total_data.shape[1]);
    
    return total_data;
#================================================================================

#================================================================================
class MRIData(Dataset):
    def __init__(self, data_files_path, transform):

        self.data_files_path = data_files_path;
        self.transform = transform;
        
        super().__init__();
    

    def __len__(self):
        return len(self.data_files_path);
    
    def __getitem__(self, index):
        mri_seg = pickle.load(open(str(self.data_files_path[index]), "rb"));
        mri = mri_seg[0];
        seg = mri_seg[1];

        mri = np.expand_dims(mri,axis = 2);
        mri = np.repeat(mri, 3, axis=2); 
        seg = np.expand_dims(seg, axis = 2);

        transformed = self.transform(image = mri, mask = seg);
        mri = transformed["image"];
        seg = transformed["mask"];

        return mri, seg;



#================================================================================