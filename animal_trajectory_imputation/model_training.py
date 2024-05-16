import copy
import datetime
import os
import numpy as np
import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim.lr_scheduler import CosineAnnealingLR
from tsl import config, logger
from tsl.data import SpatioTemporalDataModule, ImputationDataset
from tsl.data.preprocessing import StandardScaler, MinMaxScaler
from data import AnimalMovement
from tsl.nn.metrics import MaskedMetric, MaskedMAE, MaskedMSE, MaskedMRE
from tsl.utils.parser_utils import ArgParser
from tsl.utils import parser_utils, numpy_metrics
from src.models.brits import BRITS
from src.models.stat_method import MeanModel, InterpolationModel
from src.models import CsdiModel, TransformerModel
from src.imputers import BRITSImputer, MeanImputer, InterpolationImputer, CsdiImputer, TransformerImputer
from scheduler import CosineSchedulerWithRestarts
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

def parse_args(model_name='transformer', config_file='transformer.yaml', deer_id=5004):
    # Argument parser
    ########################################
    parser = ArgParser()
    parser.add_argument('--deer-id', type=int, default=deer_id)
    # parser.add_argument("--model-name", type=str, default='csdi')
    # parser.add_argument("--model-name", type=str, default='interpolation')
    parser.add_argument("--model-name", type=str, default=model_name)
    parser.add_argument("--dataset-name", type=str, default='animal_movement')
    # parser.add_argument("--config", type=str, default='csdi.yaml')
    # parser.add_argument("--config", type=str, default='interpolation.yaml')
    parser.add_argument("--config", type=str, default=config_file)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--check-val-every-n-epoch', type=int, default=1)
    parser.add_argument('--batch-inference', type=int, default=32)
    parser.add_argument('--load-from-pretrained', type=str,
                        default=None)
    ########################################

    parser.add_argument('--seed', type=int, default=-1)
    parser.add_argument('--precision', type=int, default=32)
    # Splitting/aggregation params
    parser.add_argument('--val-len', type=float, default=0.1)
    parser.add_argument('--test-len', type=float, default=0.1)
    # Training params
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--patience', type=int, default=40)
    parser.add_argument('--l2-reg', type=float, default=0)
    parser.add_argument('--batches-epoch', type=int, default=300)
    parser.add_argument('--split-batch-in', type=int, default=1)
    parser.add_argument('--grad-clip-val', type=float, default=5.)
    parser.add_argument('--loss-fn', type=str, default='l1_loss')
    parser.add_argument('--lr-scheduler', type=str, default=None)
    # Connectivity params
    parser.add_argument("--adj-threshold", type=float, default=0.1)

    parser.add_argument('--p-fault', type=float, default=0.0)
    parser.add_argument('--p-noise', type=float, default=0.1)

    known_args, _ = parser.parse_known_args()
    model_cls, imputer_cls = get_model_classes(known_args.model_name)
    parser = model_cls.add_model_specific_args(parser)
    parser = imputer_cls.add_argparse_args(parser)
    parser = SpatioTemporalDataModule.add_argparse_args(parser)
    parser = ImputationDataset.add_argparse_args(parser)

    args = parser.parse_args()
    if args.config is not None:
        cfg_path = os.path.join(config.config_dir, args.config)
        with open(cfg_path, 'r') as fp:
            config_args = yaml.load(fp, Loader=yaml.FullLoader)
        for arg in config_args:
            setattr(args, arg, config_args[arg])

    # since model is in training stage, we don't load from pretrained model
    args.load_from_pretrained = None

    return args


def get_model_classes(model_str):

    if model_str == 'brits':
        model, filler = BRITS, BRITSImputer
    elif model_str == 'interpolation':
        model, filler = InterpolationModel, InterpolationImputer
    elif model_str == 'mean':
        model, filler = MeanModel, MeanImputer
    elif model_str == 'csdi':
        model, filler = CsdiModel, CsdiImputer
    elif model_str == 'transformer':
        model, filler = TransformerModel, TransformerImputer
    else:
        raise ValueError(f'Model {model_str} not available.')
    return model, filler



def get_scheduler(scheduler_name: str = None, args=None):
    if scheduler_name is None:
        return None, None
    scheduler_name = scheduler_name.lower()
    if scheduler_name == 'cosine':
        scheduler_class = CosineAnnealingLR
        scheduler_kwargs = dict(eta_min=0.1 * args.lr, T_max=args.epochs)
    elif scheduler_name == 'magic':
        scheduler_class = CosineSchedulerWithRestarts
        scheduler_kwargs = dict(num_warmup_steps=12, min_factor=0.1,
                                linear_decay=0.67,
                                num_training_steps=args.epochs,
                                num_cycles=args.epochs // 100)

    elif scheduler_name == 'multi_step':
        scheduler_class = torch.optim.lr_scheduler.MultiStepLR
        p1 = int(0.6 * args.epochs)
        p2 = int(0.9 * args.epochs)
        scheduler_kwargs = dict(milestones=[p1, p2], gamma=0.1)
    else:
        raise ValueError(f"Invalid scheduler name: {scheduler_name}.")
    return scheduler_class, scheduler_kwargs




def run_experiment(args):
    # Set configuration and seed
    args = copy.deepcopy(args)
    if args.seed < 0:
        args.seed = np.random.randint(1e9)
    torch.set_num_threads(1)
    pl.seed_everything(args.seed)



    model_cls, imputer_class = get_model_classes(args.model_name)
    dataset = AnimalMovement(mode='train', deer_id=args.deer_id)

    # covariate dimension
    if 'covariates' in dataset.attributes:
        args.u_size = dataset.attributes['covariates'].shape[-1]
        args.covariate_dim = args.u_size
    else:
        args.u_size = 0
        args.covariate_dim = 0

    # logger.info(args)

    ########################################
    # create logdir and save configuration #
    ########################################
    exp_name = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
    exp_name = f"{exp_name}_{args.seed}"
    logdir = os.path.join(config.log_dir, args.dataset_name,
                          args.model_name, exp_name)
    # save config for logging
    os.makedirs(logdir, exist_ok=True)
    with open(os.path.join(logdir, 'config.yaml'), 'w') as fp:
        yaml.dump(parser_utils.config_dict_from_args(args), fp,
                  indent=4, sort_keys=True)

    ########################################
    # data module                          #
    ########################################

    if args.model_name in ['brits', 'csdi', 'transformer'] and 'covariates' in dataset.attributes:
        exog_map = {'covariates': dataset.attributes['covariates']}
        input_map = {
            'u': 'covariates',
            'x': 'data'
        }
    else:
        exog_map = input_map = None


    if 'st_coords' in dataset.attributes:
        if exog_map is None and input_map is None:
            exog_map = {'st_coords': dataset.attributes['st_coords']}
            input_map = {'x': 'data', 'st_coords': 'st_coords'}
        else:
            exog_map['st_coords'] = dataset.attributes['st_coords']
            input_map['st_coords'] = 'st_coords'


    # instantiate dataset
    torch_dataset = ImputationDataset(dataset.y,
                               training_mask=dataset.training_mask,
                                      eval_mask=dataset.eval_mask,
                                      exogenous=exog_map,
                                      input_map=input_map,
                                      window=args.window,
                                      stride=args.stride)


    # scalers = {'data': StandardScaler(axis=(0, 1))}
    # scalers = {'data': MinMaxScaler(axis=(0, 1), out_range=(-1, 1))}
    scalers = None

    # get train/val/test indices
    splitter = dataset.get_splitter(val_len=args.val_len,
                                    test_len=args.test_len)

    dm = SpatioTemporalDataModule(torch_dataset,
                                  scalers=scalers,
                                  splitter=splitter,
                                  batch_size=args.batch_size // args.split_batch_in)
    dm.setup()

    tmp = torch_dataset[0]

    ########################################
    # predictor                            #
    ########################################
    additional_model_hparams = dict(n_nodes=dm.n_nodes, input_size=dm.n_channels, output_size=dm.n_channels, window_size=dm.window)

    # model's inputs
    model_kwargs = parser_utils.filter_args(
        args={**vars(args), **additional_model_hparams},
        target_cls=model_cls,
        return_dict=True)

    # loss and metrics
    loss_fn = MaskedMetric(metric_fn=getattr(torch.nn.functional, args.loss_fn),
                           metric_kwargs={'reduction': 'none'})

    metrics = {'mae': MaskedMAE(compute_on_step=False),
               'mse': MaskedMSE(compute_on_step=False),
               'mre': MaskedMRE(compute_on_step=False)}

    scheduler_class, scheduler_kwargs = get_scheduler(args.lr_scheduler, args)

    # setup imputer
    imputer_kwargs = parser_utils.filter_argparse_args(args, imputer_class,
                                                       return_dict=True)
    imputer = imputer_class(
        model_class=model_cls,
        model_kwargs=model_kwargs,
        optim_class=torch.optim.Adam,
        optim_kwargs={'lr': args.lr,
                      'weight_decay': args.l2_reg},
        loss_fn=loss_fn,
        metrics=metrics,
        scheduler_class=scheduler_class,
        scheduler_kwargs=scheduler_kwargs,
        **imputer_kwargs
    )

    ########################################
    # training                             #
    ########################################

    require_training = True
    if args.model_name in ['mean', 'interpolation']:
        require_training = False

    if args.load_from_pretrained is not None:
        require_training = False

    # callbacks
    if args.loss_fn == 'l1_loss':
        monitor = 'val_mae'
    elif args.loss_fn == 'mse_loss':
        monitor = 'val_mse'

    early_stop_callback = EarlyStopping(monitor=monitor,
                                        patience=args.patience, mode='min')
    checkpoint_callback = ModelCheckpoint(dirpath=logdir, save_top_k=1,
                                          monitor=monitor, mode='min')

    tb_logger = TensorBoardLogger(logdir, name="model")

    trainer = pl.Trainer(max_epochs=args.epochs,
                         default_root_dir=logdir,
                         logger=tb_logger,
                         precision=args.precision,
                         accumulate_grad_batches=args.split_batch_in,
                         gpus=int(torch.cuda.is_available()),
                         gradient_clip_val=args.grad_clip_val,
                         limit_train_batches=args.batches_epoch * args.split_batch_in,
                         check_val_every_n_epoch=args.check_val_every_n_epoch,
                         callbacks=[early_stop_callback, checkpoint_callback])
    if require_training:
        trainer.fit(imputer,
                    train_dataloaders=dm.train_dataloader(),
                    val_dataloaders=dm.val_dataloader(
                        batch_size=args.batch_inference))
    elif args.load_from_pretrained is not None:
        imputer = imputer_class.load_from_checkpoint(args.load_from_pretrained, model_class=model_cls,
         model_kwargs=model_kwargs,
         optim_class=torch.optim.Adam,
         optim_kwargs={'lr': args.lr,
                  'weight_decay': args.l2_reg},
         loss_fn=loss_fn,
         metrics=metrics,
         scheduler_class=scheduler_class,
         scheduler_kwargs=scheduler_kwargs,
         **imputer_kwargs)


    #######################################
    # testing                              #
    #######################################
    if require_training:
        imputer.load_model(checkpoint_callback.best_model_path)
    imputer.freeze()
    trainer.test(imputer, dataloaders=dm.test_dataloader(
        batch_size=args.batch_inference))




if __name__ == '__main__':
    # make all files under Female/TagData
    deer_id_list = [int(f.split('.')[0][-4:]) for f in os.listdir('Female/TagData') if f.endswith('.csv')]

    model_list = ['csdi']

    for model in model_list:
        args = parse_args(model_name=model, config_file=f'{model}.yaml', deer_id=0)
        run_experiment(args)
        torch.cuda.empty_cache()
