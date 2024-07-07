import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch_geometric.nn import GCN
import torch.nn.functional as F
from torchvision.ops.misc import MLP


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
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
    
class GraphTransformerEncodingLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__()
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.gcn = GCN(in_channels=d_model, hidden_channels=d_model, out_channels=d_model, num_layers=1)

    def forward(self, x, edge_index, edge_weight):
        # x: (B, K, L, D)
        # edge_index: (B, 2, E)
        # edge_weight: (B, E)
        B, K, L, D = x.size()
        x = x.view(B * K, L, D)
        x = self.transformer_encoder_layer(x)
        x = x.view(B, K, L, D)


        output = x.clone()  # Create a copy of x to store the results

        for b in range(B):
            for l in range(L):         
                x_slice = x[b, :, l, :]
                x_slice = self.gcn(x_slice, edge_index[b], edge_weight[b])
                output[b, :, l, :] =  x_slice + x[b, :, l, :]

        return output

       
        


class GraphTransformer(pl.LightningModule):
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
                 weight_decay=0.0
                 ):
        super().__init__()
        self.x_dim = x_dim
        self.hidden_dims = hidden_dims
        self.lr = lr
        self.weight_decay = weight_decay
       
        self.y_enc = MLP(y_dim, hidden_dims, dropout=dropout)
        if x_dim > 0:
            self.x_enc = MLP(x_dim, hidden_dims, dropout=dropout)

        self.mask_token = nn.Parameter(torch.randn(hidden_dims[-1]))  # Learnable mask token
        self.pe = PositionalEncoding(hidden_dims[-1], dropout=dropout)

        self.encoder_layers = nn.ModuleList([
            GraphTransformerEncodingLayer(d_model=hidden_dims[-1], nhead=n_heads, dim_feedforward=ff_dim, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.readout = MLP(hidden_dims[-1], hidden_dims + [output_dim], dropout=dropout)
                                    
            
    def forward(self, y, mask, edge_index, edge_weight, x):
        # y: (B, K, L)
        # mask: (B, K, L)
        # edge_index: (B, 2, E)
        # edge_weight: (B, E)
        # x: (B, K, L, C)

        B, K, L = y.shape
        y = y * mask
        y = y.unsqueeze(-1)  # (B, K, L, 1)

        h_y = self.y_enc(y)  # (B, K, L, hidden_dims[-1])
        h_y = mask.unsqueeze(-1) * h_y + (1 - mask).unsqueeze(-1) * self.mask_token  # (B, K, L, hidden_dims[-1])


        if self.x_dim > 0:
            h_x = self.x_enc(x)  # (B, K, L, hidden_dims[-1])
            h = h_y + h_x  # (B, K, L, hidden_dims[-1])
        else:
            h = h_y

        h = self.pe(h)  # (B, K, L, hidden_dims[-1])

        for layer in self.encoder_layers:
            h = layer(h, edge_index, edge_weight)

        # Pass through the readout MLP
        x_hat = self.readout(h)  # (B, K, L, output_dim)

        return x_hat
    
    def on_train_batch_start(self, batch, batch_idx):
        mask = batch['mask']
        val_mask = batch['val_mask']
        eval_mask = batch['eval_mask']
        observed_mask = mask - eval_mask - val_mask
        p = torch.tensor([0.2, 0.5, 0.8])
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
        y_observed = y * training_mask
        y_target = y * target_mask
        edge_index = batch['edge_index']
        edge_weight = batch['edge_weight']

        if self.x_dim > 0:
            x = batch['x']
        else:
            x = None

        y_hat = self(y_observed,training_mask, edge_index, edge_weight, x)
        y_hat = y_hat.squeeze(-1)
        y_hat = y_hat * target_mask
        loss = torch.abs(y_hat - y_target).sum() / target_mask.sum()
        self.log('train_loss', loss, on_epoch=True, on_step=False, prog_bar=True)
        return loss
    
    
    
    def validation_step(self, batch, batch_idx):
        y = batch['y']
        mask = batch['mask']
        val_mask = batch['val_mask']
        eval_mask = batch['eval_mask']
        observed_mask = mask - eval_mask - val_mask
        y_observed = y * observed_mask
        y_target = y * val_mask

        edge_index = batch['edge_index']
        edge_weight = batch['edge_weight']
        
        if self.x_dim > 0:
            x = batch['x']
        else:
            x = None

        y_hat = self(y_observed, observed_mask, edge_index, edge_weight, x)
        y_hat = y_hat.squeeze(-1)
        y_hat = y_hat * val_mask
        loss = torch.abs(y_hat - y_target).sum() / val_mask.sum()
        self.log('val_loss', loss, on_epoch=True, on_step=False, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        y = batch['y']
        mask = batch['mask']
        eval_mask = batch['eval_mask']
        observed_mask = mask - eval_mask
        y_observed = y * observed_mask
        y_target = y * eval_mask
        edge_index = batch['edge_index']
        edge_weight = batch['edge_weight']
        
        if self.x_dim > 0:
            x = batch['x']
        else:
            x = None


        y_hat = self(y_observed, observed_mask, edge_index, edge_weight, x)
        y_hat = y_hat.squeeze(-1)
        y_hat = y_hat * eval_mask
        
        loss = torch.abs(y_hat - y_target).sum() / eval_mask.sum()
        self.log('test_loss', loss, on_epoch=True, on_step=False, prog_bar=True)
        return loss
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        # scheduler = {
        #     'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=20),
        #     'monitor': 'val_loss',
        #     'interval': 'epoch',
        #     'frequency': 1
        # }

        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 250], gamma=0.1)


        return [optimizer], [scheduler]


if __name__ == '__main__':
    # Define model parameters
    y_dim = 1
    x_dim = 20
    hidden_dims = [128, 128]
    output_dim = 1
    ff_dim = 128
    n_heads = 1
    n_layers = 4
    dropout = 0.0

    # Create a model instance
    model = GraphTransformer(y_dim, x_dim, hidden_dims, output_dim, ff_dim, n_heads, n_layers, dropout)

    # Generate dummy data
    B, K, L, C = 16, 36, 72, x_dim
    y = torch.randn(B, K, L)
    mask = torch.ones(B, K, L)
    edge_index = torch.randint(0, K, (B, 2, 20))
    edge_weight = torch.randn(B, 20)
    x = torch.randn(B, K, L, C)

    # Forward pass
    output = model(y, mask, edge_index, edge_weight, x)

    # Print the output shape
    print("Output shape:", output.shape)
