import torch
import pytorch_lightning as pl


def tilted_loss(y, f, q):
    e1 = y - f
    the_sum = torch.mean(torch.max(q * e1, (q - 1) * e1), axis=-1)
    return the_sum


class DNN(pl.LightningModule):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout_rate=0.0, weight_decay=0, lr=1e-3, loss_func='mse'):
        super(DNN, self).__init__()
        self.weight_decay = weight_decay
        self.lr = lr
        if loss_func == 'mse':
            self.loss_func = torch.nn.functional.mse_loss
        elif loss_func == 'mae':
            self.loss_func = torch.nn.functional.l1_loss
        elif loss_func == 'tilted':
            self.loss_func = tilted_loss

        self.layers = torch.nn.ModuleList()
        self.dropouts = torch.nn.ModuleList()
        # self.batch_norms = torch.nn.ModuleList()
        self.layers.append(torch.nn.Linear(input_dim, hidden_dims[0]))
        # self.batch_norms.append(torch.nn.BatchNorm1d(hidden_dims[0]))
        self.dropouts.append(torch.nn.Dropout(dropout_rate))
        for i in range(1, len(hidden_dims)):
            self.layers.append(torch.nn.Linear(hidden_dims[i-1], hidden_dims[i]))
            # self.batch_norms.append(torch.nn.BatchNorm1d(hidden_dims[i]))
            self.dropouts.append(torch.nn.Dropout(dropout_rate))
        self.layers.append(torch.nn.Linear(hidden_dims[-1], output_dim))
        
        # Initialize the model parameters using He initialization
        for layer in self.layers:
            torch.nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
            torch.nn.init.zeros_(layer.bias)

       
    
    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            # x = self.batch_norms[i](x)
            x = torch.relu(x)
            x = self.dropouts[i](x)
        return self.layers[-1](x).squeeze()
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_func(y_hat, y)
        
        self.log('train_loss', loss, on_epoch=True, on_step=False, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_func(y_hat, y)

        # rmse 
        rmse = torch.sqrt(torch.nn.functional.mse_loss(y_hat, y))

        self.log('val_loss', loss, on_epoch=True, on_step=False, prog_bar=True)
        self.log('val_rmse', rmse, on_epoch=True, on_step=False, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_func(y_hat, y)

        # rmse 
        rmse = torch.sqrt(torch.nn.functional.mse_loss(y_hat, y))
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

        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 250], gamma=0.1)


        return [optimizer], [scheduler]

