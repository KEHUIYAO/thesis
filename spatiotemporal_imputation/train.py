import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from torch.utils.data import random_split
from data import GP, KaustCompetition, SoilMoisture, AQ36
import argparse
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from model import DNN, DCN, GraphTransformer, Transformer
from utils import interpolate_missing_values, create_dnn_dataset, create_graph_transformer_dataset, DataModule, GraphTransformerDataModule
import yaml



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='transformer/air_quality.yaml')
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



    if args.model == 'DNN':
        train_dataset, val_dataset, test_dataset = create_dnn_dataset(st_dataset, additional_st_covariate=args.additional_st_covariate, val_ratio=args.val_ratio, test_ratio=args.test_ratio)

        # datamodule
        dm = DataModule(train_dataset, val_dataset, test_dataset, batch_size=args.batch_size)

        # model
        model = DNN(input_dim=args.input_dim, hidden_dims=args.hidden_dims, output_dim=args.output_dim, dropout_rate=args.dropout_rate, weight_decay=args.weight_decay, lr=args.lr, loss_func=args.loss_func)
    elif args.model == 'DCN':
        train_dataset, val_dataset, test_dataset = create_dnn_dataset(st_dataset, additional_st_covariate=args.additional_st_covariate, val_ratio=args.val_ratio, test_ratio=args.test_ratio)

        # datamodule
        dm = DataModule(train_dataset, val_dataset, test_dataset, batch_size=args.batch_size)

        # model
        model = DCN(input_dim=args.input_dim, cross_num=args.cross_num, dnn_hidden_units=args.dnn_hidden_units, dnn_dropout=args.dnn_dropout, weight_decay=args.weight_decay, lr=args.lr, loss_func=args.loss_func)
    
    elif args.model == 'GraphTransformer':
        dataset = create_graph_transformer_dataset(st_dataset, args.space_sigma, args.space_threshold, args.space_partitions_num, args.window_size, args.stride, args.val_ratio)
        dm = GraphTransformerDataModule(dataset, batch_size=args.batch_size)
        model = GraphTransformer(y_dim=args.y_dim, x_dim=args.x_dim, hidden_dims=args.hidden_dims, output_dim=args.output_dim, ff_dim=args.ff_dim, n_heads=args.n_heads, n_layers=args.n_layers, dropout=args.dropout, lr=args.lr, weight_decay=args.weight_decay, whiten_prob=args.whiten_prob)
    
    elif args.model == 'Transformer':
        dataset = create_graph_transformer_dataset(st_dataset, args.space_sigma, args.space_threshold, args.space_partitions_num, args.window_size, args.stride, args.val_ratio)
        dm = GraphTransformerDataModule(dataset, batch_size=args.batch_size)
        model = Transformer(input_size=args.input_size, hidden_size=args.hidden_size, output_size=args.output_size, ff_size=args.ff_size, u_size=args.u_size, n_heads=args.n_heads, n_layers=args.n_layers, dropout=args.dropout, condition_on_u=args.condition_on_u, axis=args.axis, activation=args.activation)
                
    

    # Set up the TensorBoard logger
    logger = TensorBoardLogger("tb_logs", name="my_model")

    # Set up the ModelCheckpoint callback to save the best model based on validation loss
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',  # metric to monitor
        dirpath='checkpoints/',  # directory to save the checkpoints
        filename='best-checkpoint',  # name of the saved file
        save_top_k=1,  # save only the best model
        mode='min'  # the mode for monitoring ('min' for minimizing the monitored metric)
    )

    # Set up the trainer with the checkpoint callback
    trainer = pl.Trainer(max_epochs=args.max_epochs, logger=logger, callbacks=[checkpoint_callback])

    # Train the model
    trainer.fit(model, dm)

    # Load the best checkpoint before testing
    best_model_path = checkpoint_callback.best_model_path


    if args.model == 'DNN':
        model = DNN.load_from_checkpoint(best_model_path, input_dim=args.input_dim, hidden_dims=args.hidden_dims, output_dim=args.output_dim, dropout_rate=args.dropout_rate, weight_decay=args.weight_decay, lr=args.lr, loss_func=args.loss_func)
    elif args.model == 'DCN':
        model = DCN.load_from_checkpoint(best_model_path, input_dim=args.input_dim, cross_num=args.cross_num, dnn_hidden_units=args.dnn_hidden_units, dnn_dropout=args.dnn_dropout, weight_decay=args.weight_decay, lr=args.lr, loss_func=args.loss_func)
    elif args.model == 'GraphTransformer':
        model = GraphTransformer.load_from_checkpoint(best_model_path, y_dim=args.y_dim, x_dim=args.x_dim, hidden_dims=args.hidden_dims, output_dim=args.output_dim, ff_dim=args.ff_dim, n_heads=args.n_heads, n_layers=args.n_layers, dropout=args.dropout, lr=args.lr, weight_decay=args.weight_decay, whiten_prob=args.whiten_prob)
    elif args.model == 'Transformer':
        model = Transformer.load_from_checkpoint(best_model_path, input_size=args.input_size, hidden_size=args.hidden_size, output_size=args.output_size, ff_size=args.ff_size, u_size=args.u_size, n_heads=args.n_heads, n_layers=args.n_layers, dropout=args.dropout, condition_on_u=args.condition_on_u, axis=args.axis, activation=args.activation)

    # Test the model
    trainer.test(model, dm.test_dataloader())


if __name__ == "__main__":
    args = get_args()
    main(args)