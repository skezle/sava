import scipy
import time
import copy
import torch

from otdd.pytorch.distance_fast import DatasetDistance, FeatureCost

import numpy as np

from visualise import visualize_values_distr_sorted, log_values_sorted
from data import get_indices


def get_per_batch_OT_cost(
    feature_extractor,
    x_tr,
    y_tr,
    x_val,
    y_val,
    batch_size=64,
    p=2,
    resize=32,
    classes=torch.arange(start=0, end=10),
    device="cuda",
    label_distances=None,
    feat_repr=False,
    parallel=False,
    cuda_num=0,
    n_gpu=8,
):
    embedder = feature_extractor.to(device)

    if feat_repr:
        embedder.linear = torch.nn.Identity()
    else:
        embedder.fc = torch.nn.Identity() # doesn't exist so this do nothing

    for param in embedder.parameters():
        param.requires_grad = False
    
    if parallel:
        embedder = torch.nn.DataParallel(
            embedder,
            device_ids=list(range(cuda_num, cuda_num + n_gpu)),
        )

    # Here we use same embedder for both datasets
    feature_cost = FeatureCost(
        src_embedding=embedder,
        src_dim=(3, resize, resize),
        tgt_embedding=embedder,
        tgt_dim=(3, resize, resize),
        p=2,
        device=device,
    )

    dist = DatasetDistance(
        (x_tr, y_tr, classes),
        (x_val, y_val, classes),
        inner_ot_method="exact",
        debiased_loss=True,
        feature_cost=feature_cost,
        λ_x=1.0,
        λ_y=1.0,
        sqrt_method="spectral",
        sqrt_niters=10,
        precision="single",
        p=p,
        entreg=1e-1,
        device=device,
        min_labelcount=1,
        pre_computed_label_dist=label_distances,
    )

    tic = time.perf_counter()
    # costs \in 1, |x_tr|, |x_val|
    costs, label_distances = dist.dual_sol_costs(maxsamples=batch_size, return_couplings=True)

    dual_sol = dist.dual_sol(maxsamples=batch_size) # n_tr, n_val

    # plan \in |x_tr|, |x_val|
    plan = dist.compute_coupling()

    toc = time.perf_counter()
    print(f"per batch cost calculation takes {toc - tic:0.4f} seconds")

    return costs.to("cpu").numpy(), plan, dual_sol, label_distances
    
# Get dual solution of OT problem
# def get_batch_OT_dual_sol(
#     feature_extractor,
#     x_tr,
#     y_tr,
#     x_val,
#     y_val,
#     batch_size=64,
#     p=2,
#     resize=32,
#     classes=torch.arange(start=0, end=10),
#     device="cuda",
# ):
#     embedder = feature_extractor.to(device)
#     embedder.fc = torch.nn.Identity()
#     for param in embedder.parameters():
#         param.requires_grad = False

#     # Here we use same embedder for both datasets
#     feature_cost = FeatureCost(
#         src_embedding=embedder,
#         src_dim=(3, resize, resize),
#         tgt_embedding=embedder,
#         tgt_dim=(3, resize, resize),
#         p=p,
#         device=device,
#     )

#     dist = DatasetDistance(
#         (x_tr, y_tr, classes),
#         (x_val, y_val, classes),
#         inner_ot_method="exact",
#         debiased_loss=True,
#         feature_cost=feature_cost,
#         λ_x=1.0,
#         λ_y=1.0,
#         sqrt_method="spectral",
#         sqrt_niters=10,
#         precision="single",
#         p=p,
#         entreg=1e-1,
#         device=device,
#         min_labelcount=2,
#     )

#     tic = time.perf_counter()
#     dual_sol = dist.dual_sol(maxsamples=batch_size) # n_tr, n_val

#     toc = time.perf_counter()
#     print(f"distance calculation takes {toc - tic:0.4f} seconds")

#     for i in range(len(dual_sol)):
#         dual_sol[i] = dual_sol[i].to("cpu")
#     return dual_sol

# Get dual solution of OT problem
def get_OT_dual_sol(
    feature_extractor,
    trainloader,
    testloader,
    training_size=10000,
    p=2,
    resize=32,
    feat_repr=False,
    device="cuda",
):
    embedder = feature_extractor.to(device)
    if feat_repr:
        embedder.linear = torch.nn.Identity()
    else:
        embedder.fc = torch.nn.Identity() # doesn't exist so this do nothing
    for param in embedder.parameters():
        param.requires_grad = False

    # Here we use same embedder for both datasets
    feature_cost = FeatureCost(
        src_embedding=embedder,
        src_dim=(3, resize, resize),
        tgt_embedding=embedder,
        tgt_dim=(3, resize, resize),
        p=p,
        device=device,
    )

    dist = DatasetDistance(
        trainloader,
        testloader,
        inner_ot_method="exact",
        debiased_loss=True,
        feature_cost=feature_cost,
        λ_x=1.0,
        λ_y=1.0,
        sqrt_method="spectral",
        sqrt_niters=10,
        precision="single",
        p=p,
        entreg=1e-1,
        device=device,
        min_labelcount=2,
    )

    tic = time.perf_counter()
    dual_sol = dist.dual_sol(maxsamples=training_size, return_coupling=True) # n_tr, n_val

    toc = time.perf_counter()
    print(f"distance calculation takes {toc - tic:0.4f} seconds")

    for i in range(len(dual_sol)):
        dual_sol[i] = dual_sol[i].to("cpu")
    return dual_sol



def train_with_corrupt_flag(trainloader, shuffle_ind, train_indices):
    trained_with_flag = []
    itr = 0
    counting_labels = {}  # For statistics
    for trai in trainloader:
        # print(trai)
        train_images = trai[0]
        train_labels = trai[1]
        # get one image of the training from that batch
        for i in range(len(train_labels)):
            train_image = train_images[i]
            train_label = train_labels[i]
            trained_with_flag.append(
                [train_image, train_label, train_indices[itr] in shuffle_ind]
            )
            itr = itr + 1
            if train_label.item() in counting_labels:
                counting_labels[train_label.item()] += 1
            else:
                counting_labels[train_label.item()] = 1
    return trained_with_flag # List[(img, label, bool)]


def compute_dual(
    feature_extractor,
    trainloader,
    testloader,
    training_size,
    shuffle_ind,
    p=2,
    resize=32,
    feat_repr=False,
    device="cuda",
):
    # to return 2
    # get indices of corrupted and non corrupted for visualization
    train_indices = get_indices(trainloader)
    # list of tuples: (img, class, {T, F})
    trained_with_flag = train_with_corrupt_flag(trainloader, shuffle_ind, train_indices)

    # to return 1
    # OT Dual calculation
    dual_sol = get_OT_dual_sol(
        feature_extractor, 
        trainloader=trainloader, 
        testloader=testloader,
        training_size=training_size,
        p=2, 
        resize=32, 
        feat_repr=feat_repr,
        device=device,
    )
    return dual_sol, trained_with_flag

def compute_costs(
    feature_extractor,
    trainloader,
    testloader,
    training_size,
    shuffle_ind,
    p=2,
    resize=32,
    device="cuda",
):
    # OT Dual calculation
    costs = get_per_batch_OT_cost(
        feature_extractor, trainloader, testloader, p=2, resize=32, device=device
    )
    return costs

# Get the calibrated gradient of the dual solution
# Which can be considered as data values (more in paper...)
def get_calibrated_gradients(dual_sol, training_size):
    f1k = np.array(dual_sol[0].squeeze().cpu()) # dual_sol: tuple len(2) dual_sol[0] = [tr_size, 1] dual
    train_gradient = [0] * training_size
    train_gradient = (1 + 1 / (training_size - 1)) * f1k - sum(f1k) / (training_size - 1)
    return list(train_gradient)

# Get the data values and also visualizes the detection of 'bad' data
def compute_values_and_visualize(
    dual_sol, 
    trained_with_flag, 
    training_size, 
    portion, 
    tag="",
):
    calibrated_gradient = get_calibrated_gradients(dual_sol, training_size) # len n_tr
    assert len(calibrated_gradient) == training_size
    sorted_gradient_ind = sort_and_keep_indices(calibrated_gradient, training_size)
    assert len(sorted_gradient_ind) == training_size
    log_values_sorted(
        trained_with_flag,
        sorted_gradient_ind,
        training_size,
        portion,
        tag=tag,
    )

    return sorted_gradient_ind, trained_with_flag

def compute_values_half_det_rate(dual_sol, trained_with_flag, training_size, portion):
    calibrated_gradient = get_calibrated_gradients(dual_sol, training_size) # len n_tr
    assert len(calibrated_gradient) == training_size
    sorted_gradient_ind = sort_and_keep_indices(calibrated_gradient, training_size)
    assert len(sorted_gradient_ind) == training_size

    poisoned = training_size * portion
    half_tr_sz = int(training_size / 2)

    # Selecting the corruption tag from the ordered list of indexes - which have been ordered according to the OT
    found = sum([trained_with_flag[sorted_gradient_ind[i][0]][2] for i in range(half_tr_sz)])
    print(
        f"inspected: {half_tr_sz}, found: {found} detection rate: {found / poisoned:.2f} baseline: {half_tr_sz*0.2*0.9}"
    )
    return found


# Sort the calibrated values and keep original indices
def sort_and_keep_indices(train_gradient, training_size, asc=False):
    orig_train_gradient = copy.deepcopy(train_gradient)
    if isinstance(train_gradient, np.ndarray):
        if asc:
            # lower value is ordered first
            sorted_gradient_ind = np.argsort(train_gradient)
        else:
            # higher value is ordered first
            sorted_gradient_ind = np.argsort(train_gradient)[::-1]
        sorted_gradient_ind = [np.array([x]) for x in sorted_gradient_ind]
    else:
        train_gradient.sort(reverse=True)
        sorted_gradient_ind = [
            np.where(orig_train_gradient == train_gradient[i])[0] for i in range(training_size)
        ]
    return sorted_gradient_ind


def dual_lp(a, b, C, verbose=0):
    """Solves the dual optimal transport problem:
    max <a, alpha> + <b, beta> s.t. alpha_i + beta_j <= C_{i,j}
    """
    m = len(a)
    n = len(b)

    c = np.concatenate((a, b))
    c *= -1  # maximization problem

    # Build alpha_i + beta_j <= C_{i,j} constraints.
    A = np.zeros((m * n, m + n))
    b = np.zeros(m * n)

    idx = 0
    for i in range(m):
        for j in range(n):
            A[idx, i] = 1
            A[idx, m + j] = 1
            b[idx] = C[i, j]
            idx += 1

    # Needs this equality constraint to make the problem bounded.
    A_eq = np.zeros((1, m + n))
    b_eq = np.zeros(1)
    A_eq[0, :m] = 1

    res = scipy.optimize.linprog(c, A, b, A_eq, b_eq, bounds=(None, None))

    if verbose:
        print("success:", res.success)
        print("status:", res.status)

    alpha = res.x[:m]
    beta = res.x[m:]

    return alpha, beta

