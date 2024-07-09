from torch import nn
from tsl.nn.blocks.encoders.mlp import MLP
from tsl.nn.blocks.encoders.transformer import SpatioTemporalTransformerLayer, \
    TransformerLayer
from tsl.nn.layers import PositionalEncoding
import torch
from tsl.nn.layers.norm import LayerNorm
from utils import CosineSchedulerWithRestarts
import pytorch_lightning as pl

class Transformer(pl.LightningModule):
    r"""Spatiotemporal Transformer for multivariate time series imputation.

    Args:
        input_size (int): Input size.
        hidden_size (int): Dimension of the learned representations.
        output_size (int): Dimension of the output.
        ff_size (int): Units in the MLP after self attention.
        u_size (int): Dimension of the exogenous variables.
        n_heads (int, optional): Number of parallel attention heads.
        n_layers (int, optional): Number of layers.
        dropout (float, optional): Dropout probability.
        axis (str, optional): Dimension on which to apply attention to update
            the representations.
        activation (str, optional): Activation function.
    """

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 output_size: int,
                 ff_size: int,
                 u_size: int,
                 n_heads: int = 1,
                 n_layers: int = 1,
                 dropout: float = 0.,
                 condition_on_u: bool = True,
                 axis: str = 'both',
                 activation: str = 'elu',
                 ):
        super().__init__()
        self.whiten_prob = [0.2, 0.5, 0.8]

        self.dummy  = nn.Linear(input_size, output_size)

        self.condition_on_u = condition_on_u
        if condition_on_u:
            self.u_enc = MLP(u_size, hidden_size, n_layers=2)
        self.h_enc = MLP(input_size, hidden_size, n_layers=2)

        self.mask_token = nn.Parameter(torch.randn(hidden_size))  # Learnable mask token

        self.pe = PositionalEncoding(hidden_size)


        kwargs = dict(input_size=hidden_size,
                      hidden_size=hidden_size,
                      ff_size=ff_size,
                      n_heads=n_heads,
                      activation=activation,
                      causal=False,
                      dropout=dropout)

        if axis in ['steps', 'nodes']:
            transformer_layer = TransformerLayer
            kwargs['axis'] = axis
        elif axis == 'both':
            transformer_layer = SpatioTemporalTransformerLayer
        else:
            raise ValueError(f'"{axis}" is not a valid axis.')

        self.encoder = nn.ModuleList()
        self.readout = nn.ModuleList()
        self.layer_norm = nn.ModuleList()
        for _ in range(n_layers):
            self.encoder.append(transformer_layer(**kwargs))
            self.layer_norm.append(LayerNorm(hidden_size))
            self.readout.append(MLP(input_size=hidden_size,
                                    hidden_size=ff_size,
                                    output_size=output_size,
                                    n_layers=2,
                                    dropout=dropout))

    def forward(self, y, mask, adj, x, **kwargs):

        # y = y.unsqueeze(-1)

        # y = self.dummy(y)

        if x is not None:
            u = x.clone()
        else:
            u = None

        x = y.clone()

        # x: [batches steps nodes features]
        # u: [batches steps (nodes) features]
        
        
        x = x.permute(0, 2, 1)
        mask = mask.permute(0, 2, 1)
        x = x.unsqueeze(-1)
        mask = mask.unsqueeze(-1)

        if u is not None:
            u = u.permute(0, 2, 1, 3)


        x = x * mask

        

        h = self.h_enc(x)
        

        h = mask * h + (1 - mask) * self.mask_token

        if self.condition_on_u and u is not None:
            h = h + self.u_enc(u)

        h = self.pe(h)


        # space encoding
        B, L, K, C = h.shape


        out = []
        for encoder, mlp, layer_norm in zip(self.encoder, self.readout, self.layer_norm):
            h = encoder(h)
            h = layer_norm(h)
            out.append(mlp(h))

        x_hat = out.pop(-1)
        x_hat = x_hat.permute(0, 2, 1, 3)
        return x_hat


    
    def on_train_batch_start(self, batch, batch_idx):
        mask = batch['mask']
        val_mask = batch['val_mask']
        eval_mask = batch['eval_mask']
        observed_mask = mask - eval_mask - val_mask
        p = torch.tensor(self.whiten_prob)
        p_size = [mask.size(0)] + [1] * (mask.ndim - 1)

        p = p[torch.randint(len(p), p_size)].to(device=mask.device)

        whiten_mask = torch.rand(mask.size(), device=mask.device) < p
        whiten_mask = whiten_mask.float()
        batch['training_mask'] = observed_mask * (1 - whiten_mask)
        batch['target_mask'] = observed_mask * whiten_mask

        batch['y_train'] = batch['y'] * batch['training_mask']
        batch['y_target'] = batch['y'] * batch['target_mask']

    

    def training_step(self, batch, batch_idx):
        y = batch['y']
        training_mask = batch['training_mask']
        target_mask = batch['target_mask']

        if target_mask.sum() == 0:
            return None

        y_observed = y * training_mask
        y_target = y * target_mask
        adj = batch['adj']

        if self.condition_on_u:
            x = batch['x']
        else:
            x = None

        y_hat = self(y_observed,training_mask, adj, x)
        y_hat = y_hat.squeeze(-1)
        y_hat = y_hat * target_mask
        # loss = torch.abs(y_hat - y_target).sum() / target_mask.sum()
        
        mean = batch['mean']
        std = batch['std']
        y_hat = y_hat * std + mean
        y_target = y_target * std + mean
        loss = torch.abs(y_hat - y_target).sum() / target_mask.sum()
        self.log('train_loss', loss, on_epoch=True, on_step=False, prog_bar=True)
        return loss

    # def training_step(self, batch, batch_idx):
    #     y = batch['y']
    #     mask = batch['mask']
    #     val_mask = batch['val_mask']

    #     # Check if val_mask.sum() == 0 and skip the batch if true
    #     if val_mask.sum() == 0:
    #         return None


    #     eval_mask = batch['eval_mask']
    #     observed_mask = mask - eval_mask - val_mask
    #     y_observed = y * observed_mask
    #     y_target = y * val_mask

    #     adj = batch['adj']
        
    #     if self.x_dim > 0:
    #         x = batch['x']
    #     else:
    #         x = None

    #     y_hat = self(y_observed, observed_mask, adj, x)
    #     y_hat = y_hat.squeeze(-1)
    #     y_hat = y_hat * val_mask
    #     loss = torch.abs(y_hat - y_target).sum() / val_mask.sum()

    #     mean = batch['mean']
    #     std = batch['std']
    #     y_hat = y_hat * std + mean
    #     y_target = y_target * std + mean
    #     train_loss = torch.abs(y_hat - y_target).sum() / val_mask.sum()
    #     self.log('train_loss', train_loss, on_epoch=True, on_step=False, prog_bar=True)
    #     return loss
    
    
    
    def validation_step(self, batch, batch_idx):
        y = batch['y']
        mask = batch['mask']
        val_mask = batch['val_mask']

        # Check if val_mask.sum() == 0 and skip the batch if true
        if val_mask.sum() == 0:
            return None


        eval_mask = batch['eval_mask']
        observed_mask = mask - eval_mask - val_mask
        y_observed = y * observed_mask
        y_target = y * val_mask

        adj = batch['adj']
        
        if self.condition_on_u:
            x = batch['x']
        else:
            x = None

        y_hat = self(y_observed, observed_mask, adj, x)
        y_hat = y_hat.squeeze(-1)
        y_hat = y_hat * val_mask
        # loss = torch.abs(y_hat - y_target).sum() / val_mask.sum()

        mean = batch['mean']
        std = batch['std']
        y_hat = y_hat * std + mean
        y_target = y_target * std + mean
        loss = torch.abs(y_hat - y_target).sum() / val_mask.sum()


        self.log('val_loss', loss, on_epoch=True, on_step=False, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        y = batch['y']
        mask = batch['mask']
        eval_mask = batch['eval_mask']

        if eval_mask.sum() == 0:
            return None



        observed_mask = mask - eval_mask
        y_observed = y * observed_mask
        y_target = y * eval_mask

        adj = batch['adj']
        
        if self.condition_on_u:
            x = batch['x']
        else:
            x = None


        y_hat = self(y_observed, observed_mask, adj, x)
        y_hat = y_hat.squeeze(-1)
        y_hat = y_hat * eval_mask
        
        # loss = torch.abs(y_hat - y_target).sum() / eval_mask.sum()

        mean = batch['mean']
        std = batch['std']
        y_hat = y_hat * std + mean
        y_target = y_target * std + mean
        loss = torch.abs(y_hat - y_target).sum() / eval_mask.sum()

        self.log('test_loss', loss, on_epoch=True, on_step=False, prog_bar=True)
        return loss
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=0.0)
        # scheduler = {
        #     'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=20),
        #     'monitor': 'val_loss',
        #     'interval': 'epoch',
        #     'frequency': 1
        # }

        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 50], gamma=0.1)

        
        # scheduler = CosineSchedulerWithRestarts(
        #     optimizer,
        #     num_warmup_steps=12,
        #     min_factor=0.1,
        #     linear_decay=0.67,
        #     num_training_steps=300,
        #     num_cycles=300 // 100
        # )

        return [optimizer], [scheduler]


