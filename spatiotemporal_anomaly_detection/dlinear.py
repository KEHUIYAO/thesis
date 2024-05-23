import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class Model(nn.Module):
    """
    Decomposition-Linear
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        # Decompsition Kernel Size
        kernel_size = 25
        self.decompsition = series_decomp(kernel_size)
        self.individual = configs.individual
        self.channels = configs.enc_in

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()

            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.seq_len, self.pred_len))

                # Use this two lines if you want to visualize the weights
                # self.Linear_Seasonal[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
                # self.Linear_Trend[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

            # Use this two lines if you want to visualize the weights
            # self.Linear_Seasonal.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
            # self.Linear_Trend.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
                                          dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0), trend_init.size(1), self.pred_len],
                                       dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        x = seasonal_output + trend_output
        return x.permute(0, 2, 1)  # to [Batch, Output length, Channel]


class SyntheticMultichannelDataset(Dataset):
    def __init__(self, input_len, pred_len, num_samples, num_channels):
        self.input_len = input_len
        self.pred_len = pred_len
        self.seq_len = input_len + pred_len
        self.num_samples = num_samples
        self.num_channels = num_channels
        self.data = self._generate_data()

    def _generate_data(self):
        data = []
        for _ in range(self.num_samples):
            sample = []
            for _ in range(self.num_channels):
                x = np.linspace(0, 10, self.seq_len)
                y = np.sin(x) + np.random.normal(0, 0.1, self.seq_len)
                sample.append(y)
            data.append(np.array(sample))
        return np.array(data)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x = self.data[idx, :, :self.input_len]
        y = self.data[idx, :, -self.pred_len:]
        return torch.tensor(x, dtype=torch.float32).transpose(0, 1), torch.tensor(y, dtype=torch.float32).transpose(0, 1)


class LightningModel(pl.LightningModule):
    def __init__(self, model):
        super(LightningModel, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)

    def predict_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        return y_hat

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.mse_loss(y_hat, y)
        self.log('train_loss', loss,  prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=1e-3)




class STData(Dataset):
    def __init__(self, df, input_len, pred_len, stride):
        self.df = df
        self.input_len = input_len
        self.pred_len = pred_len
        self.stride = stride

        self.data = self._generate_data()

    def _generate_data(self):
        C, L = self.df.shape
        segments = []

        for start_idx in range(0, L - self.input_len - self.pred_len + 1, self.stride):
            input_segment = self.df[:, start_idx:start_idx + self.input_len]
            pred_segment = self.df[:, start_idx + self.input_len:start_idx + self.input_len + self.pred_len]
            segments.append((input_segment, pred_segment))

        return segments

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        return torch.tensor(x, dtype=torch.float32).transpose(0, 1), torch.tensor(y, dtype=torch.float32).transpose(0,1)



class Configs:
    def __init__(self, seq_len, pred_len, individual, enc_in):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.individual = individual
        self.enc_in = enc_in


if __name__ == '__main__':
    input_len = 5
    pred_len = 1
    num_channels = 400
    # stride = 1
    stride = pred_len
    individual = True
    max_epochs = 1


    # dataset = SyntheticMultichannelDataset(input_len, pred_len, num_samples, num_channels)
    df = np.random.rand(num_channels, 100)
    dataset = STData(df, input_len, pred_len, stride=stride)
    print(len(dataset))
    training_dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    configs = Configs(input_len, pred_len, individual=individual, enc_in=num_channels)
    model = Model(configs)
    lightning_model = LightningModel(model)

    trainer = pl.Trainer(max_epochs=max_epochs)
    trainer.fit(lightning_model, training_dataloader)

    # Generate predictions
    predictions = trainer.predict(lightning_model, test_dataloader)

    # concatenate predictions along the first dimension
    predictions = torch.cat(predictions, dim=0).squeeze().detach().numpy()

    # transpose the predictions to [num_channels, pred_len]
    predictions = predictions.transpose(1, 0)

    print(predictions)
