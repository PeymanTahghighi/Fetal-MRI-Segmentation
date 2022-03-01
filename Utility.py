import numpy as np
import matplotlib.pyplot as plt
import cv2
from collections import Counter
import torch

def window_center_adjustment(img, max_val):

    hist = np.histogram(img.ravel(), bins = int(max_val))[0];
    hist = hist / hist.sum();
    hist = np.cumsum(hist);

    hist_thresh = ((1-hist) < 5e-4);
    max_intensity = np.where(hist_thresh == True)[0][0];
    adjusted_img = img * (255/(max_intensity + 1e-4));
    adjusted_img = np.where(adjusted_img > 255, 255, adjusted_img).astype("uint8");

    return adjusted_img;

def evaluation_metrics(stats_scores):
    tp = stats_scores[:,0];
    fp = stats_scores[:,1];
    tn = stats_scores[:,2];
    fn = stats_scores[:,3];

    prec = tp/(tp + fp + 1e-6);
    rec = tp/(tp + fn + 1e-6);
    f1 = (2*prec*rec)/(prec+rec+ 1e-6);
    vs = 1 - (torch.abs(fn - fp))/(2*tp + fp+ fn);

    return torch.mean(prec), torch.mean(rec), torch.mean(f1), torch.mean(vs);
