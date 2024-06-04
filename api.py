import lava
import ot
import pickle
import os
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data.dataloader as dataloader

import visualise

def lava_experiment(
    feature_extractor: nn.Module,
    train_loader: dataloader.DataLoader,
    val_loader: dataloader.DataLoader,
    training_size: int,
    shuffle_ind: int,
    resize: int,
    portion: float,
    feat_repr: bool,
    device: torch.device,
    tag: str = "",
):
    # dual_sol is a tuple element 0 is of shape (1, tr_size) element 1 is of shape (1, val_size)
    dual_sol, trained_with_flag = lava.compute_dual(
        feature_extractor,
        train_loader,
        val_loader,
        training_size,
        shuffle_ind,
        resize=resize,
        feat_repr=feat_repr,
        device=device,
    )

    sorted_gradient_ind, trained_with_flag = lava.compute_values_and_visualize(
        dual_sol, trained_with_flag, training_size, portion, tag,
    )
    return sorted_gradient_ind, trained_with_flag

def batchwise_lava_experiment(
    feature_extractor: nn.Module,
    train_loader: dataloader.DataLoader,
    val_loader: dataloader.DataLoader,
    training_size: int,
    batch_size: int,
    shuffle_ind: int,
    resize: int,
    portion: float,
    feat_repr: bool,
    device: torch.device,
    cache_label_distances: bool,
    tag: str = "",
    num_classes: int = 10,
    parallel: bool = False,
    cuda_num: int = 0,
    n_gpu: int = 8,
):
    values = []
    label_distances = None
    for i, (x_tr, y_tr) in enumerate(tqdm(train_loader, desc="SAVA valuation")):
        values_arr_tmp = np.zeros(x_tr.shape[0])
        for j, (x_val, y_val) in enumerate(val_loader):  
            # cost \in (tr_batch_size, val_batch_size)
            _, _, dual_sol, label_distances = lava.get_per_batch_OT_cost(
                feature_extractor, 
                x_tr.reshape(x_tr.shape[0], -1), 
                y_tr,
                x_val.reshape(x_val.shape[0], -1),
                y_val,
                batch_size=batch_size,
                p=2, 
                resize=resize,
                classes=torch.arange(start=0, end=num_classes),
                device=device,
                label_distances=label_distances if cache_label_distances else None,
                feat_repr=feat_repr,
                parallel=parallel,
                cuda_num=cuda_num,
                n_gpu=n_gpu,
            )
            assert dual_sol[0].shape[1] == x_tr.shape[0]
            assert dual_sol[1].shape[1] == x_val.shape[0]
            calibrated_gradient = lava.get_calibrated_gradients(
                dual_sol, 
                training_size=x_tr.shape[0], 
            )
            # Apply tanh to squash values to the range (-1, 1), then scale and translate to (0, 1)
            squashed_calibrated_gradient = (np.tanh(calibrated_gradient) + 1) / 2
            # Normalize the array so its sum equals 1
            values_arr_tmp += squashed_calibrated_gradient / squashed_calibrated_gradient.sum()
        values.append(values_arr_tmp / len(val_loader))

    values = np.concatenate(values)
    sorted_gradient_ind = lava.sort_and_keep_indices(train_gradient=values, training_size=min(training_size, len(values)))
    # for Clothing1M experiments we don't know a priori
    # which instances are noisy, so we don't have portion and 
    # shuffle_ind variables nor can we calculate a detection
    # rate. Let's just return values.
    if portion is None and shuffle_ind is None:
        return sorted_gradient_ind
    else:
        trained_indices = lava.get_indices(train_loader)
        trained_with_flag = lava.train_with_corrupt_flag(train_loader, shuffle_ind, trained_indices) # len training set
        
        visualise.log_values_sorted(
            trained_with_flag,
            sorted_gradient_ind,
            min(training_size, len(values)), # see comment above
            portion,
            tag=tag,
        )
        return sorted_gradient_ind, trained_with_flag

def hierarchical_ot_experiment(
    feature_extractor: nn.Module,
    train_loader: dataloader.DataLoader,
    val_loader: dataloader.DataLoader,
    training_size: int,
    batch_size: int,
    shuffle_ind: np.array,
    resize: int,
    portion: float,
    device: torch.device,
    cache_label_distances: bool,
    visualise_hot: bool = False,
    tag: str = "",
    feat_repr: bool = False,
    num_classes: int = 10,
    parallel: bool = False,
    cuda_num: int = 0,
    n_gpu: int = 8,
):

    dual_sol_dict = {i: {} for i in range(len(train_loader))}
    costs_bar = np.zeros((len(train_loader), len(val_loader)))

    label_distances = None
    for i, (x_tr, y_tr) in enumerate(tqdm(train_loader, desc="SAVA valuation")):
        for j, (x_val, y_val) in enumerate(val_loader):  
                # cost and plan are (tr_batch_size, val_batch_size)
                cost, plan, dual_sol, label_distances = lava.get_per_batch_OT_cost(
                    feature_extractor, 
                    x_tr.reshape(x_tr.shape[0], -1), 
                    y_tr,
                    x_val.reshape(x_val.shape[0], -1),
                    y_val,
                    batch_size=batch_size,
                    p=2, 
                    resize=resize,
                    classes=torch.arange(start=0, end=num_classes),
                    device=device,
                    label_distances=label_distances if cache_label_distances else None,
                    feat_repr=feat_repr,
                    parallel=parallel,
                    cuda_num=cuda_num,
                    n_gpu=n_gpu,
                )
                costs_bar[i, j] = 1 / (x_tr.shape[0] + x_val.shape[0]) * np.sum(plan * cost)
                assert dual_sol[0].shape[1] == x_tr.shape[0]
                assert dual_sol[1].shape[1] == x_val.shape[0]
                dual_sol_dict[i][j] = dual_sol

    # line 7: compute dual sol on \bar{C}
    a = np.ones(costs_bar.shape[0]) # vector 1, dimension = row of barC
    b = np.ones(costs_bar.shape[1]) # vector 1, dimension = column of barC
    eps = np.max(costs_bar)
    #f_bar, g_bar = lava.dual_lp(a, b, costs_bar / eps) # (num_train_batches, ) (num_val_batces, )
    #plan_bar = (np.eye(costs_bar.shape[0]) * np.squeeze(f_bar)) @ np.exp(-costs_bar / eps) @ (np.eye(costs_bar.shape[1]) * np.squeeze(g_bar)) # (num_tr_batches, num_val_batches)
    plan_bar = ot.sinkhorn(a, b, costs_bar / eps, 1e-02, verbose=False)
    if visualise_hot:
        cache_dict = {
            'cost_batches': costs_bar,
            'plan_batches': plan_bar,
            'cost_final_batch': cost,
            'plan_final_batch': plan,
        }
        filename = os.path.join(
            os.getcwd(), 
            'output',
            "sava_artifacts.pickle"
        )
        with open(filename, 'wb') as file:  # 'wb' indicates that you are writing in binary mode
        # Pickle the dictionary and write it to the file
            pickle.dump(cache_dict, file)
        
    # important data point selection
    values = []
    # iterate over train batches
    for k, (x_tr, y_tr) in enumerate(train_loader):
        # iterate over each point in the batch
        for l in range(x_tr.shape[0]):
            threshold_gradients = np.zeros((len(val_loader)))
            # iterate over all the val batches
            for m in range(len(val_loader)):
            
                dual_sol = dual_sol_dict[k][m]
                # len x_tr.shape[0]
                calibrated_gradient = lava.get_calibrated_gradients(
                    dual_sol, 
                    training_size=x_tr.shape[0], 
                )
            
                threshold_gradients[m] = calibrated_gradient[l]

            # line 13 in Alg1
            s_l = np.sum(plan_bar[k] * threshold_gradients)

            values.append(s_l)
    
    # training size is min(training_size, len(values)) since for the poison frogs
    # corrution some of the perturned points don;t meet a certain requirement and are dropped
    # to the final training set from the datasets.py might not be the same size as is specified in the 
    # training_size variable
    sorted_gradient_ind = lava.sort_and_keep_indices(train_gradient=values, training_size=min(training_size, len(values)))
    
    # for Clothing1M experiments we don't know a priori
    # which instances are noisy, so we don't have portion and 
    # shuffle_ind variables nor can we calculate a detection
    # rate. Let's just return values.
    if portion is None and shuffle_ind is None:
        return sorted_gradient_ind
    else:
        trained_indices = lava.get_indices(train_loader)
        trained_with_flag = lava.train_with_corrupt_flag(train_loader, shuffle_ind, trained_indices) # len training set

        visualise.log_values_sorted(
            trained_with_flag,
            sorted_gradient_ind,
            min(training_size, len(values)), # see comment above
            portion,
            tag=tag,
        )

        return sorted_gradient_ind, trained_with_flag
     
