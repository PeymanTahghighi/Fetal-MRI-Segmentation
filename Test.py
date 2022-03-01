import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import cv2
from collections import Counter

def window_center_adjustment(img, max_val):

    hist = np.histogram(img.ravel(), bins= max_val)[0];
    hist = hist / hist.sum();
    hist = np.cumsum(hist);

    hist_thresh = ((1-hist) < 5e-4);
    max_intensity = np.where(hist_thresh == True)[0][0];
    adjusted_img = img * (255/max_intensity);
    adjusted_img = np.where(adjusted_img > 255, 255, adjusted_img).astype("uint8");

    return adjusted_img;


if __name__ == "__main__":
    root_path_img = "D:\PhD\Courses\Image Analysis\Project\\Dataset\\feta_2.1\sub-006\\anat\\sub-006_rec-mial_T2w.nii.gz";
    root_path_mask = "D:\PhD\Courses\Image Analysis\Project\\Dataset\\feta_2.1\sub-048\\anat\\sub-048_rec-mial_dseg.nii.gz";

    img = nib.load(root_path_img).get_fdata();
    window_center_adjustment(img[129,:,:]);

    #for i in range(256):
    
    
    plt.show();

