#================================================================
#================================================================
from copy import deepcopy
from copyreg import pickle
import os
from glob import glob
from django import conf
from sklearn.model_selection import KFold
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn
import torch
import pickle
from torchmetrics import Precision, Recall, F1, StatScores
import Config
from Dataset import MRIData, dataset_slicer, load_entire_MRI
from TrainHandler import eval_one_epoch, train_one_epoch
from Network import Unet2D
#================================================================
#================================================================

if __name__ == "__main__":

    torch.autograd.set_detect_anomaly(False)
    torch.autograd.profiler.profile(False)
    torch.autograd.profiler.emit_nvtx(False)
    torch.backends.cudnn.benchmark = True
    
    #dataset_slicer(root_path="D:\PhD\Courses\Image Analysis\Project\Dataset\Pure", output_root="D:\PhD\Courses\Image Analysis\Project\Dataset\\2D slice");
    entire_dataset = np.array(glob("D:\PhD\Courses\Image Analysis\Project\Dataset\\2D slice\\*"));

    #for k-fold cross-validation
    kfold = KFold(n_splits=5);

    #initialize required variables
    model = Unet2D(8).to(Config.DEVICE);
    optimizer = optim.Adam(model.parameters(), lr = Config.LEARNING_RATE);
    loss_func = nn.CrossEntropyLoss();
    scaler = torch.cuda.amp.grad_scaler.GradScaler()

    # total_data = [];
    # for train,test in kfold.split(entire_dataset):
    #     total_data.append([train,test]);
    
    # pickle.dump(total_data, open("fold_data.fd", 'wb'));

    #define estimator of the model
    stat_scores = StatScores(num_classes=8, reduce='macro').to(Config.DEVICE);

    #for holding fold number
    f = 4;

    #load fold from file
    folds = pickle.load(open("fold_data.fd", "rb"));
    train = folds[4][0];
    test = folds[4][1];
    #for train,test in kfold.split(entire_dataset):
    print(f"\n==========================================================================\n\
        Starting fold {f}...\
            \n==========================================================================");

    #Get data and generate data loaders for this batch
    train_folders = np.take(entire_dataset,train);
    test_folders = np.take(entire_dataset,test);

    train_data = load_entire_MRI(train_folders);
    test_data = load_entire_MRI(test_folders);

    train_loader = DataLoader(MRIData(train_data,
                                Config.train_transforms),
                                batch_size=Config.BATCH_SIZE,
                                num_workers = Config.NUM_WORKERS,
                                pin_memory=True);

    test_loader = DataLoader(MRIData(test_data,
                                Config.valid_transforms),
                                batch_size=Config.BATCH_SIZE,
                                num_workers = Config.NUM_WORKERS,
                                pin_memory=True);
    #---------------------------------------------------------------

    #reset model parameters each time we want to run a new fold
    model.reset_weights();

    #epoch counter
    e = 1;
    #for early stopping
    best_loss = 10;
    best_prec = 0;
    best_rec = 0;
    best_f1 = 0;
    best_vs = 0;
    best_each_class_metrics = 0;
    best_model_weights = None;
    patience = Config.EARLY_STOPPING_PATIENCE;
    while(True):

        train_one_epoch(model, train_loader, loss_func, optimizer, scaler);

        total_train_loss, prec_train, rec_train, f1_train, vs_train, _ = eval_one_epoch(model, 
        train_loader, 
        loss_func,
        stat_scores);

        total_test_loss, prec_test, rec_test, f1_test, vs_test, each_class_metrics = eval_one_epoch(model, 
        test_loader, 
        loss_func,
        stat_scores);

        print(f"Epoch {e}\tTrain:\nLoss: {total_train_loss}\t\
            Precision: {prec_train} \
            Recall: {rec_train}\t\
            F1: {f1_train}\t\
            VS: {vs_train}");
        
        print(f"Test:\nLoss: {total_test_loss}\t\
            Precision: {prec_test} \
            Recall: {rec_test}\t\
            F1: {f1_test}\t\
            VS: {vs_test}");
        
        e += 1;

        #if we have found a better model, save the results
        if total_test_loss < best_loss:
            print("New best model found...!");
            patience = Config.EARLY_STOPPING_PATIENCE;
            best_loss = total_test_loss;
            best_prec = prec_test;
            best_rec = rec_test;
            best_f1 = f1_test;
            best_vs = vs_test;
            best_each_class_metrics = each_class_metrics;
            best_model_weights = deepcopy(model.state_dict);
        
        #if we haven't found a better model, decrease patience
        patience -= 1;
        
        #if we ran out of patience, exit the loop
        #if patience == 0:
        break;
    
    print(f"Fold {f}:\nLoss: {best_loss}\t\
            Precision: {best_prec} \
            Recall: {best_rec}\t\
            F1: {best_f1}\t\
            VS: {best_vs}");
    
    print(best_each_class_metrics.detach().cpu().numpy());

    #save best model weights
    pickle.dump(best_model_weights, open(f"model-CE-2D-f{f}.mdl", "wb"));

    f += 1;
