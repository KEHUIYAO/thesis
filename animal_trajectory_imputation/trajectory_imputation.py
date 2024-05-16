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
    dataset = AnimalMovement(mode='imputation', deer_id=args.deer_id)

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
    scalers = {'data': MinMaxScaler(axis=(0, 1), out_range=(-1, 1))}

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
    # trainer.test(imputer, dataloaders=dm.test_dataloader(
    #     batch_size=args.batch_inference))


    ########################################
    # testing                              #
    ########################################
    if args.model_name in ['csdi', 'diffgrin']:
        enable_multiple_imputation = True
    else:
        enable_multiple_imputation = False


    dataset = AnimalMovement(mode='imputation', deer_id=args.deer_id)
    # scaler = StandardScaler(axis=(0, 1))
    # scaler = MinMaxScaler(axis=(0, 1), out_range=(-1, 1))
    # scaler.fit(dataset.y, dataset.training_mask)
    # scaler.bias = torch.tensor(scaler.bias)
    # scaler.scale = torch.tensor(scaler.scale)
    # scalers = {'data': scaler}
    scalers = None

    # instantiate dataset
    if enable_multiple_imputation:
        stride = int(args.window / 2)
    else:
        stride = args.stride
    torch_dataset = ImputationDataset(dataset.y,
                                      training_mask=dataset.training_mask,
                                      eval_mask=dataset.eval_mask,
                                      exogenous=exog_map,
                                      input_map=input_map,
                                      window=args.window,
                                      stride=stride,
                                      scalers=scalers)

    # get train/val/test indices
    splitter = dataset.get_splitter(val_len=0,
                                    test_len=len(torch_dataset))

    dm = SpatioTemporalDataModule(torch_dataset,
                                  splitter=splitter,
                                  batch_size=args.batch_size // args.split_batch_in)
    dm.setup()

    y_hat = []
    y_true = []
    eval_mask = []
    observed_mask = []
    st_coords = []



    if enable_multiple_imputation:
        multiple_imputations = []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for batch_id, batch in enumerate(
            tqdm(dm.test_dataloader(batch_size=args.batch_inference), desc="Processing", leave=True)):

        batch = batch.to(device)
        imputer = imputer.to(device)

        batch = imputer.on_after_batch_transfer(batch, 0)
        output = imputer.predict_step(batch, batch_id)

        y_hat.append(output['y_hat'].detach().cpu().numpy())
        y_true.append(output['y'].detach().cpu().numpy())
        eval_mask.append(output['eval_mask'].detach().cpu().numpy())
        observed_mask.append(output['observed_mask'].detach().cpu().numpy())
        st_coords.append(output['st_coords'].detach().cpu().numpy())
        if enable_multiple_imputation:
            multiple_imputations.append(output['imputed_samples'].detach().cpu().numpy())

    y_hat = np.concatenate(y_hat, axis=0)
    y_true = np.concatenate(y_true, axis=0)
    eval_mask = np.concatenate(eval_mask, axis=0)
    observed_mask = np.concatenate(observed_mask, axis=0)
    st_coords = np.concatenate(st_coords, axis=0)



    if enable_multiple_imputation:
        multiple_imputations = np.concatenate(multiple_imputations, axis=0)


    seq_len = dataset.y.shape[0]
    num_nodes = dataset.y.shape[1]
    C = 2
    y_true_original = np.zeros([seq_len, num_nodes, C])
    y_hat_original = np.zeros([seq_len, num_nodes, C])
    if enable_multiple_imputation:
        y_hat_multiple_imputation = np.zeros([multiple_imputations.shape[1], seq_len, num_nodes, C])
        count_multiple_imputation = np.zeros([multiple_imputations.shape[1], seq_len, num_nodes, C])
    observed_mask_original = np.zeros([seq_len, num_nodes, C])
    eval_mask_original = np.zeros([seq_len, num_nodes, C])
    count = np.zeros([seq_len, num_nodes, C])


    B, L, K, C = y_hat.shape
    for b in range(B):
        for l in range(L):
            for k in range(K):
                ts_pos = st_coords[b, l, k, ::-1]
                y_hat_original[ts_pos[0], ts_pos[1]] = y_hat_original[ts_pos[0], ts_pos[1]] + y_hat[b, l, k]
                count[ts_pos[0], ts_pos[1]] = count[ts_pos[0], ts_pos[1]] + 1

                observed_mask_original[ts_pos[0], ts_pos[1]] = observed_mask[b, l, k]
                eval_mask_original[ts_pos[0], ts_pos[1]] = eval_mask[b, l, k]

                if enable_multiple_imputation:
                    y_hat_multiple_imputation[:, ts_pos[0], ts_pos[1]] = y_hat_multiple_imputation[:, ts_pos[0], ts_pos[1]] + multiple_imputations[b, :, l, k]
                    count_multiple_imputation[:, ts_pos[0], ts_pos[1]] = count_multiple_imputation[:, ts_pos[0], ts_pos[1]] + 1

    # for those positions that count is not 0, we divide the sum by count to get the average
    y_hat_original[count != 0] = y_hat_original[count != 0] / count[count != 0]

    if enable_multiple_imputation:
        y_hat_multiple_imputation[count_multiple_imputation != 0] = y_hat_multiple_imputation[count_multiple_imputation != 0] / count_multiple_imputation[count_multiple_imputation != 0]



    # use hold-out test-set
    y_true_original = dataset.y
    jul = dataset.attributes['covariates'][:, 0, 0]
    eval_mask_original = dataset.eval_mask

    eval_mask_original = eval_mask_original & (count != 0)


    # number of observed data points
    n_observed = np.sum(observed_mask_original)



    # save output to file
    output = {}
    output['y_hat'] = y_hat_original
    output['y'] = y_true_original
    output['eval_mask'] = eval_mask_original
    output['observed_mask'] = observed_mask_original

    if enable_multiple_imputation:
        output['imputed_samples'] = y_hat_multiple_imputation

    # create a folder called results/deer_id and save the result
    if not os.path.exists(f'./results/{args.deer_id}/{args.model_name}'):
        os.makedirs(f'./results/{args.deer_id}/{args.model_name}')

    # save it to a npz file
    np.savez(f'./results/{args.deer_id}/{args.model_name}/output.npz', **output)


    # plot the result and save it
    all_target_np = y_true_original.squeeze(-2)
    all_evalpoint_np = eval_mask_original.squeeze(-2)
    all_observed_np = observed_mask_original.squeeze(-2)
    if enable_multiple_imputation:
        samples = y_hat_multiple_imputation.squeeze(-2)
    else:
        samples = y_hat_original.squeeze(-2)[np.newaxis, ...]
    qlist = [0.05, 0.25, 0.5, 0.75, 0.95]
    quantiles_imp = []
    for q in qlist:
        tmp = np.quantile(samples, q, axis=0)
        quantiles_imp.append(tmp * (1 - all_observed_np) + all_target_np * all_observed_np)


    #######################################
    offset = 72
    B = 5
    plt.rcParams["font.size"] = 16

    # set seed 42
    rng = np.random.default_rng(42)

    for i in range(B):

        fig, axes = plt.subplots(nrows=C, ncols=1, figsize=(36, 24.0))

        start = offset
        end = all_target_np.shape[0] - offset

        # randomly pick a time period of 100 steps between start and end
        start = rng.choice(np.arange(start, end - 100))
        end = start + 100

        for k in range(C):
            df = pd.DataFrame(
                {"x": jul[start:end], "val": all_target_np[start:end, k], "y": all_evalpoint_np[start:end, k]})
            df = df[df.y != 0]
            df2 = pd.DataFrame(
                {"x": jul[start:end], "val": all_target_np[start:end, k], "y": all_observed_np[start:end, k]})
            df2 = df2[df2.y != 0]
            # axes[k].plot(jul[start:end], quantiles_imp[2][start:end, k], color='g', linestyle='solid', label='CSDI')
            axes[k].plot(jul[start:end], quantiles_imp[2][start:end, k], color='g', marker='o', label='CSDI',)
            axes[k].fill_between(jul[start:end], quantiles_imp[0][start:end, k], quantiles_imp[4][start:end, k],
                                 color='g', alpha=0.3)
            axes[k].plot(df2.x, df2.val, color='r', marker='x', linestyle='None')
            axes[k].plot(df.x, df.val, color='b', marker='o', linestyle='None')

        # save the plot
        plt.savefig(f'./results/{args.deer_id}/{args.model_name}/prediction{i}.png')

        plt.close()





if __name__ == '__main__':
    # make all files under Female/TagData
    deer_id_list = sorted([int(f.split('.')[0][-4:]) for f in os.listdir('Female/TagData') if f.endswith('.csv')])

    # randomly select 20% of the deer ids as testing data
    rng = np.random.RandomState(42)
    rng.shuffle(deer_id_list)
    # deer_id_list = deer_id_list[int(0.8 * len(deer_id_list)):]
    deer_id_list = [5094]



    model_list = ['interpolation', 'csdi']

    # deer_id_list = [5629, 5631, 5633, 5639, 5657]
    # deer_id_list = [5000, 5004, 5006, 5016,5022,5037, 5043]
    for i in deer_id_list:
        for model in model_list:
            args = parse_args(model_name=model, config_file=f'{model}.yaml', deer_id=i)

            print('Running deer_id:', i, 'model:', model)

            # try:
            #     run_experiment(args)
            # except:
            #     pass
            run_experiment(args)

            torch.cuda.empty_cache()

