import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from torch.utils.data import random_split
from data import GP, KaustCompetition, SoilMoisture, AQ36, AQ
import argparse
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging
from model import DNN, DCN, SpatialTemporalTransformer
from utils import interpolate_missing_values, create_dnn_dataset, SpatialTemporalTransformerDataset, DataModule, SpatialTemporalTransformerDataModule
import yaml



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='st_transformer/kaust_competition_st_basis.yaml')
    args = parser.parse_args()
    config_file = './experiment/' + args.config
    with open(config_file, 'r') as fp:
        config_args = yaml.load(fp, Loader=yaml.FullLoader)
    for arg in config_args:
        setattr(args, arg, config_args[arg])

    return args


def main(args):
    if args.dataset == 'GP':
        st_dataset = GP(args.num_nodes, args.seq_len)
    elif args.dataset == 'KaustCompetition':
        st_dataset = KaustCompetition(args.dataset_index)
    elif args.dataset == 'SoilMoisture':
        st_dataset = SoilMoisture()
    elif args.dataset == 'AirQuality':
        st_dataset = AQ36()
    elif args.dataset == 'AQ':
        st_dataset = AQ()
    else:
        raise ValueError('Invalid dataset name')
    
    



    if args.model == 'DNN':
        train_dataset, val_dataset, test_dataset = create_dnn_dataset(st_dataset, additional_st_covariate=args.additional_st_covariate, val_ratio=args.val_ratio)

        # datamodule
        dm = DataModule(train_dataset, val_dataset, test_dataset, batch_size=args.batch_size)

        # model
        model = DNN(input_dim=args.input_dim, hidden_dims=args.hidden_dims, output_dim=args.output_dim, dropout_rate=args.dropout_rate, weight_decay=args.weight_decay, lr=args.lr, loss_func=args.loss_func)
    elif args.model == 'DCN':
        train_dataset, val_dataset, test_dataset = create_dnn_dataset(st_dataset, additional_st_covariate=args.additional_st_covariate, val_ratio=args.val_ratio)

        # datamodule
        dm = DataModule(train_dataset, val_dataset, test_dataset, batch_size=args.batch_size)

        # model
        model = DCN(input_dim=args.input_dim, cross_num=args.cross_num, dnn_hidden_units=args.dnn_hidden_units, dnn_dropout=args.dnn_dropout, weight_decay=args.weight_decay, lr=args.lr, loss_func=args.loss_func)
    
    elif args.model == 'SpatialTemporalTransformer':
        y, x, mask, eval_mask, space_coords, time_coords = st_dataset.y, st_dataset.x, st_dataset.mask, st_dataset.eval_mask, st_dataset.space_coords, st_dataset.time_coords
        dataset = SpatialTemporalTransformerDataset(y, x, mask, eval_mask, space_coords, time_coords, args.correlation_threshold, args.space_partitions_num, args.window_size, args.stride, args.val_ratio, args.additional_st_covariates, args.normalization_axis, args.training_strategy)
        dm = SpatialTemporalTransformerDataModule(dataset, batch_size=args.batch_size)
        model = SpatialTemporalTransformer(y_dim=args.y_dim, x_dim=args.x_dim, hidden_dims=args.hidden_dims, output_dim=args.output_dim, ff_dim=args.ff_dim, n_heads=args.n_heads, n_layers=args.n_layers, dropout=args.dropout, lr=args.lr, weight_decay=args.weight_decay, whiten_prob=args.whiten_prob, training_strategy=args.training_strategy)
  
    

    # Set up the TensorBoard logger
    logger = TensorBoardLogger("tb_logs", name="my_model")

    # Set up the ModelCheckpoint callback to save the best model based on validation loss
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',  # metric to monitor
        dirpath=f'checkpoints/{args.dataset}/{args.model}',  # directory to save the checkpoints
        filename='best-checkpoint',  # name of the saved file
        save_top_k=1,  # save only the best model
        mode='min'  # the mode for monitoring ('min' for minimizing the monitored metric)
    )

    # Set up the trainer with the checkpoint callback
    if args.model == 'SpatialTemporalTransformer':
        trainer = pl.Trainer(max_epochs=args.max_epochs, logger=logger, callbacks=[checkpoint_callback, StochasticWeightAveraging(swa_lrs=1e-2)], reload_dataloaders_every_n_epochs=1, precision="16-mixed")
    else:
        trainer = pl.Trainer(max_epochs=args.max_epochs, logger=logger, callbacks=[checkpoint_callback])


    # Train the model
    trainer.fit(model, dm)

   

    # Load the best checkpoint before testing
    best_model_path = checkpoint_callback.best_model_path


    if args.model == 'DNN':
        model = DNN.load_from_checkpoint(best_model_path, input_dim=args.input_dim, hidden_dims=args.hidden_dims, output_dim=args.output_dim, dropout_rate=args.dropout_rate, weight_decay=args.weight_decay, lr=args.lr, loss_func=args.loss_func)
    elif args.model == 'DCN':
        model = DCN.load_from_checkpoint(best_model_path, input_dim=args.input_dim, cross_num=args.cross_num, dnn_hidden_units=args.dnn_hidden_units, dnn_dropout=args.dnn_dropout, weight_decay=args.weight_decay, lr=args.lr, loss_func=args.loss_func)
    elif args.model == 'SpatialTemporalTransformer':
        model = SpatialTemporalTransformer.load_from_checkpoint(best_model_path, y_dim=args.y_dim, x_dim=args.x_dim, hidden_dims=args.hidden_dims, output_dim=args.output_dim, ff_dim=args.ff_dim, n_heads=args.n_heads, n_layers=args.n_layers, dropout=args.dropout, lr=args.lr, weight_decay=args.weight_decay, whiten_prob=args.whiten_prob)
   
    # Test the model
    trainer.test(model, dm.test_dataloader())


if __name__ == "__main__":
    args = get_args()
    main(args)