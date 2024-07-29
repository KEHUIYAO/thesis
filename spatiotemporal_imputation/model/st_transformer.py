import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from torchvision.ops.misc import MLP
from utils import CosineSchedulerWithRestarts
from torch_geometric.nn.conv import GATConv



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.0, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        B, K, L, D = x.size()
        pe = self.pe[:L, :].unsqueeze(0).unsqueeze(0)  # (1, 1, L, D)
        x = x + pe
        return self.dropout(x)
    

class SpatialTemporalTransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__()
        self.temporal_transformer_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True, norm_first=True, activation='gelu')
        self.spatial_transformer_layer = GATConv(d_model, d_model, heads=nhead, dropout=dropout)


    def forward(self, x, edge_index):
        # x: (B, K, L, D)
        B, K, L, D = x.size()
        
        # Apply temporal transformer layer
        x = x.reshape(B * K, L, D)
        x = self.temporal_transformer_layer(x)
        x = x.reshape(B, K, L, D)
        
        # Apply spatial transformer layer
        res = torch.zeros_like(x).to(x.device)
        for l in range(L):
            res[0, :, l, :] = self.spatial_transformer_layer(x[0, :, l, :], edge_index[0, :, :])
        
        return res


    

        


class SpatialTemporalTransformer(pl.LightningModule):
    def __init__(self,
                 y_dim,
                 x_dim,
                 hidden_dims,
                 output_dim,
                 ff_dim,
                 n_heads,
                 n_layers,
                 dropout,
                 lr=1e-3,
                 weight_decay=0.0,
                 whiten_prob=[0.2, 0.5, 0.8],
                 loss_func='mae'
                 ):
        super().__init__()
        self.x_dim = x_dim
        self.hidden_dims = hidden_dims
        self.lr = lr
        self.weight_decay = weight_decay
        self.whiten_prob = whiten_prob
        self.loss_func = loss_func
       
        self.y_enc = MLP(y_dim, hidden_dims, dropout=dropout)
        self.layer_norm_y = nn.LayerNorm(hidden_dims[-1])

        if x_dim > 0:
            self.x_enc = MLP(x_dim, hidden_dims, dropout=dropout)
            self.readin = MLP(hidden_dims[-1]+hidden_dims[-1], hidden_dims, dropout=dropout)
            self.layer_norm_x = nn.LayerNorm(hidden_dims[-1])

        self.mask_token = nn.Parameter(torch.randn(hidden_dims[-1]))  # Learnable mask token
        self.pe = PositionalEncoding(hidden_dims[-1], dropout=dropout)

        self.encoder_layers = nn.ModuleList([
            SpatialTemporalTransformerLayer(d_model=hidden_dims[-1], nhead=n_heads, dim_feedforward=ff_dim, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.readout = MLP(hidden_dims[-1], hidden_dims + [output_dim], dropout=dropout)
                                    
            
    def forward(self, y, mask, x, edge_index):
        # y: (B, K, L)
        # mask: (B, K, L)
        # x: (B, K, L, C)
        # adj: (B, K, K)

        B, K, L = y.shape
        y = y * mask
        y = y.unsqueeze(-1)  # (B, K, L, 1)

        h_y = self.y_enc(y)  # (B, K, L, hidden_dims[-1])
        h_y = mask.unsqueeze(-1) * h_y + (1 - mask).unsqueeze(-1) * self.mask_token  # (B, K, L, hidden_dims[-1])
        h_y = mask.unsqueeze(-1) * h_y
        # h_y = self.layer_norm_y(h_y)


        if self.x_dim > 0:
            h_x = self.x_enc(x)  # (B, K, L, hidden_dims[-1])
            # h_x = self.layer_norm_x(h_x)
            h = torch.cat([h_y, h_x], dim=-1)  # (B, K, L, 2 * hidden_dims[-1])
            h = self.readin(h)
        else:
            h = h_y

        h = self.pe(h)  # (B, K, L, hidden_dims[-1])

        for layer in self.encoder_layers:
            h = layer(h, edge_index)

        # Pass through the readout MLP
        x_hat = self.readout(h)  # (B, K, L, output_dim)

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
        edge_index = batch['edge_index']
        training_mask = batch['training_mask']
        target_mask = batch['target_mask']

        if target_mask.sum() == 0:
            return None

        y_observed = y * training_mask
        y_target = y * target_mask
        

        if self.x_dim > 0:
            x = batch['x']
        else:
            x = None

        y_hat = self(y_observed,training_mask, x, edge_index)
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

    
    
    
    def validation_step(self, batch, batch_idx):
        y = batch['y']
        edge_index = batch['edge_index']
        mask = batch['mask']
        val_mask = batch['val_mask']

        # Check if val_mask.sum() == 0 and skip the batch if true
        if val_mask.sum() == 0:
            return None


        eval_mask = batch['eval_mask']
        observed_mask = mask - eval_mask - val_mask
        y_observed = y * observed_mask
        y_target = y * val_mask

        
        if self.x_dim > 0:
            x = batch['x']
        else:
            x = None

        y_hat = self(y_observed, observed_mask, x, edge_index)
        y_hat = y_hat.squeeze(-1)
        y_hat = y_hat * val_mask
        # loss = torch.abs(y_hat - y_target).sum() / val_mask.sum()

        mean = batch['mean']
        std = batch['std']
        y_hat = y_hat * std + mean
        y_target = y_target * std + mean
        loss = torch.abs(y_hat - y_target).sum() / val_mask.sum()

        rmse = torch.sqrt(F.mse_loss(y_hat, y_target, reduction='sum') / val_mask.sum())


        self.log('val_loss', loss, on_epoch=True, on_step=False, prog_bar=True)
        self.log('val_rmse', rmse, on_epoch=True, on_step=False, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        y = batch['y']
        edge_index = batch['edge_index']
        mask = batch['mask']
        eval_mask = batch['eval_mask']

        if eval_mask.sum() == 0:
            return None



        observed_mask = mask - eval_mask
        y_observed = y * observed_mask
        y_target = y * eval_mask

     
        
        if self.x_dim > 0:
            x = batch['x']
        else:
            x = None


        y_hat = self(y_observed, observed_mask, x, edge_index)
        y_hat = y_hat.squeeze(-1)
        y_hat = y_hat * eval_mask
        
        # loss = torch.abs(y_hat - y_target).sum() / eval_mask.sum()

        mean = batch['mean']
        std = batch['std']
        y_hat = y_hat * std + mean
        y_target = y_target * std + mean
        loss = torch.abs(y_hat - y_target).sum() / eval_mask.sum()

        rmse = torch.sqrt(F.mse_loss(y_hat, y_target, reduction='sum') / eval_mask.sum())

        self.log('test_loss', loss, on_epoch=True, on_step=False, prog_bar=True)
        self.log('test_rmse', rmse, on_epoch=True, on_step=False, prog_bar=True)
        return loss
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        # scheduler = {
        #     'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=20),
        #     'monitor': 'val_loss',
        #     'interval': 'epoch',
        #     'frequency': 1
        # }

        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 50], gamma=0.1)

        
        scheduler = CosineSchedulerWithRestarts(
            optimizer,
            num_warmup_steps=12,
            min_factor=0.1,
            linear_decay=0.67,
            num_training_steps=300,
            num_cycles=3
        )

        return [optimizer], [scheduler]



# class GraphConvLayer(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(GraphConvLayer, self).__init__()
#         self.linear = nn.Linear(in_channels, out_channels)

#     def forward(self, x, adj):
#         # x: (B, K, L, D)
#         # adj: (B, K, K)
#         B, K, L, D = x.size()
        
        
#         # Matrix multiplication over all temporal slices at once
#         x = x.permute(0, 2, 1, 3)  # (B, L, K, D)
#         x = torch.einsum('bkk,blkd->blkd', adj, x)  # (B, L, K, D)
#         x = x.permute(0, 2, 1, 3)  # (B, K, L, D)
        
#         # Apply linear layer
#         out = self.linear(x)
        
#         return out

# class GCN(nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels):
#         super(GCN, self).__init__()
#         self.conv1 = GraphConvLayer(in_channels, hidden_channels)
#         self.layer_norm1 = nn.LayerNorm(hidden_channels)
#         self.relu = nn.ReLU()
#         self.conv2 = GraphConvLayer(hidden_channels, out_channels)
#         self.layer_norm2 = nn.LayerNorm(out_channels)


#     def forward(self, x, adj):
#         # x: (B, K, L, D)
#         # adj: (B, K, K)
#         B, K, L, D = x.size()
        
#         # Create the adjacency matrix for each batch element
#         x = self.conv1(x, adj)
#         x = self.layer_norm1(x)
#         x = self.relu(x)
#         x = self.conv2(x, adj)
#         x = self.layer_norm2(x)
#         return x
    
