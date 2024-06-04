import argparse

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--cuda_num', 
        type=int, 
        default=0,
    )
    parser.add_argument(
        '--hierarchical', 
        action='store_true',
        default=False, 
        help='whether to train with the hierarchical OT algo',
    )
    parser.add_argument(
        '--batchwise_lava', 
        action='store_true',
        default=False, 
        help='whether to train with an independent LAVA run per train/val batch.',
    )
    parser.add_argument(
        '--random_seed', 
        type=int, 
        default=2021,
    )
    parser.add_argument(
        '--tag', 
        type=str,
        default='', 
        help='unique tag',
    )
    parser.add_argument(
        '--corruption_type',
        required=True, 
        type=str,
        help='corruption type, in {shuffle, feature, trojan_sq, poison_frogs}',
    )
    parser.add_argument(
        '--cache_l2l', 
        action='store_true',
        default=False, 
        help='whether cache label-2-label distances.',
    )
    parser.add_argument(
        '--hot_batch_size',
        type=int,
        default=1024,
        help='hierarchical ot batch size.',
    )
    parser.add_argument(
        '--corrupt_por',
        type=float,
        default=0.0,
        help='fraction of corrupted data.',
    )
    parser.add_argument(
        '--prune_perc',
        type=float,
        default=0.1,
        help='fraction of data remove from the dataset.',
    )
    parser.add_argument(
        '--smoketest', 
        action='store_true',
        default=False, 
        help='whether to run a tesr run.',
    )
    parser.add_argument(
        '--disable_wandb', 
        action='store_true',
        default=False, 
        help='whether to stop wandb logging.',
    )
    parser.add_argument(
        '--train_net', 
        action='store_true',
        default=False, 
        help='whether to run training of a model with pruned dataset.',
    )
    parser.add_argument(
        '--feat_repr', 
        action='store_true',
        default=False, 
        help='whether to use the feat dim or the output dim for LAVA.',
    )
    parser.add_argument(
        '--poison_frogs_feat_repr', 
        action='store_true',
        default=False,
        help='whether to use the feat dim or the output dim repr for poison frogs attack.',
    )
    parser.add_argument(
        '--train_dataset_sizes',  
        nargs="+", 
        type=int, 
        required=False, 
        default = [500, 1000, 2000, 5000, 10000, 20000, 50000],
        help='training set sizes to loop over.',
    )
    parser.add_argument(
        '--remake_data',
        action='store_true',
        default=False,
        help='whether to reset the data creation process for poison frogs attack.',
    )
    parser.add_argument(
        '--cache_tag', 
        type=str,
        default='', 
        help='unique tag',
    )
    parser.add_argument(
        '--data_gen_force_cpu',
        action='store_true',
        default=False,
        help='whether to use the cpu for data gen only useful for avoiding poison frogs seg faults.',
    )
    parser.add_argument(
        '--stratified',
        action='store_true',
        default=False, 
        help='whether to train stratified sampling (each class is sampled evenly within a batch).',
    )
    parser.add_argument(
        '--visualise_hot',
        action='store_true',
        default=False, 
        help='whether to cache artifacts from HOT/SAVA algorithm.',
    )
    parser.add_argument(
        '--val_dataset_size',  
        type=int,  
        default = 10000,
        help='val/test set size.',
    )
    parser.add_argument(
        '--evaluate',  
        action='store_true',
        default=False, 
        help='whether to run on the test set versus the valid.',
    )
    args = parser.parse_known_args(args=args)[0]
    return args

def nonstat_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--cuda_num', 
        type=int, 
        default=0,
    )
    parser.add_argument(
        '--hierarchical', 
        action='store_true',
        default=False, 
        help='whether to train with the hierarchical OT algo',
    )
    parser.add_argument(
        '--random_seed', 
        type=int, 
        default=2021,
    )
    parser.add_argument(
        '--tag', 
        type=str,
        default='', 
        help='unique tag',
    )
    parser.add_argument(
        '--corruption_type',
        required=True, 
        type=str,
        help='corruption type, in {shuffle, feature, trojan_sq, poison_frogs}',
    )
    parser.add_argument(
        '--cache_l2l', 
        action='store_true',
        default=False, 
        help='whether cache label-2-label distances.',
    )
    parser.add_argument(
        '--hot_batch_size',
        type=int,
        default=1024,
        help='hierarchical ot batch size.',
    )
    parser.add_argument(
        '--corrupt_por',
        type=float,
        default=0.0,
        help='fraction of corrupted data.',
    )
    parser.add_argument(
        '--prune_perc',
        type=float,
        default=0.1,
        help='fraction of data remove from the dataset.',
    )
    parser.add_argument(
        '--smoketest', 
        action='store_true',
        default=False, 
        help='whether to run a tesr run.',
    )
    parser.add_argument(
        '--resume', 
        action='store_true',
        default=False, 
        help='whether resume the non stationary loop',
    )
    parser.add_argument(
        '--resume_inds_path', 
        type=str, 
        default='output/indices/',
        help='path to the resume inds',
    )
    parser.add_argument(
        '--resume_checkpoint_path',
        type=str, 
        default='output/checkpoint/',
        help='path to the resume inds',
    )
    parser.add_argument(
        '--resume_epoch',
        type=int,
        default=50,
        help='epoch to resume, needs corresponding model checkpoint.',
    )
    parser.add_argument(
        '--knn_sv', 
        action='store_true',
        default=False, 
        help='whether to run knn sv valuation and pruning.',
    )
    parser.add_argument(
        '--k',
        type=int,
        default=10,
    )
    parser.add_argument(
        '--output_repr',
        action='store_true',
        default=False, 
        help='whether to use the feature space or the output repr.',
    )
    parser.add_argument(
        '--val_dataset_size',  
        type=int,  
        default=10000,
        help='val set size.',
    )
    args = parser.parse_known_args(args=args)[0]
    return args

def parse_args_clothing1m(args=None):
    parser = argparse.ArgumentParser(description='PyTorch Clothing1M')
    parser.add_argument(
        '--seed',
        type=int,
        default=2021,
    )
    parser.add_argument(
        '--cuda_num',
        type=int, 
        help='number of cuda in the server',
    )
    parser.add_argument(
        '--n_gpu', 
        type=int,
        default=2, 
        help='number of gpu to use',
    )
    parser.add_argument(
        '--root', 
        type=str, 
        default='data/clothing1M/',
        help='root of dataset',
    )
    parser.add_argument(
        '--value_batch_size',
        type=int,
        default=1024,
        help='hierarchical ot / EL2N batch size.',
    )
    parser.add_argument(
        '--train_batch_size',
        type=int,
        default=256,
        help='training batch size.',
    )
    parser.add_argument(
        '--tag', 
        type=str,
        default='', 
        help='unique tag',
    )
    parser.add_argument(
        '--disable_wandb', 
        action='store_true',
        default=False, 
        help='whether to disable wandb logging - useful for smoketest.',
    )
    parser.add_argument(
        '--smoketest', 
        action='store_true',
        default=False, 
        help='whether to run the scrtip in smoketest mode.',
    )
    parser.add_argument(
        '--hot', 
        action='store_true',
        default=False, 
        help='whether to run the scrtip with hot/sava data valuation or random pruning.',
    )
    parser.add_argument(
        '--el2n', 
        action='store_true',
        default=False, 
        help='whether to run the scrtip with e2ln data valuation.',
    )
    parser.add_argument(
        '--el2n_value_model_seed',
        type=int,
        default=10,
        help='init seed of the el2n model, which are used for valuation.',
    )
    parser.add_argument(
        '--el2n_value_num_models',
        type=int,
        default=10,
        help='num el2n models to do mc expectation over weights.',
    )
    parser.add_argument(
        '--prune_percs',  
        nargs="+", 
        type=float, 
        required=False, 
        default = [0.0, 0.1, 0.2, 0.3, 0.4],
        help='pruning percentages to loop over.',
    )
    parser.add_argument(
        '--preact_resnet', 
        action='store_true', 
        default=False, 
        help='Train a preact resnet instead of normal resnet.',
    )
    parser.add_argument(
        '--wd',
        type=float, 
        default=5e-4,
        help='weight decay coef.',
    )
    parser.add_argument(
        '--prune_interval', 
        type=str,
        default='right', 
        help='For EL2N, defines the pruning interval of the vlaues.',
    )
    parser.add_argument(
        '--values_tag', 
        type=str,
        default='', 
        help='Filename SAVA/EL2N/SLP values.',
    )
    parser.add_argument(
        '--slp',
        action='store_true',
        default=False, 
        help='whether to run the scrtip with supervised prototypes value pruning.',
    )
    parser.add_argument(
        '--batch_lava',
        action='store_true',
        default=False, 
        help='whether to run the scrtip with batch-wise LAVA value pruning.',
    )
    args = parser.parse_known_args(args=args)[0]
    return args
