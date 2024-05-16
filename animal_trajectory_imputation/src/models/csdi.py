import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import torch
import torch.nn as nn




def get_torch_trans(heads=8, layers=1, channels=64):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu"
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)


def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer


class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim / 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)  # (T,1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)  # (1,dim)
        table = steps * frequencies  # (T,dim)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2)
        return table



class ResidualBlock(nn.Module):
    def __init__(self, cond_dim, hidden_dim, diffusion_embedding_dim, nheads):
        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, hidden_dim)
        self.cond_projection = Conv1d_with_init(cond_dim, 2 * hidden_dim, 1)
        self.mid_projection = Conv1d_with_init(hidden_dim, 2 * hidden_dim, 1)
        self.output_projection = Conv1d_with_init(hidden_dim, 2 * hidden_dim, 1)

        self.time_layer = get_torch_trans(heads=nheads, layers=1, channels=hidden_dim)
        self.feature_layer = get_torch_trans(heads=nheads, layers=1, channels=hidden_dim)






    def forward_time(self, y, base_shape):
        B, channel, K, L = base_shape
        if L == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)
        y = self.time_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)
        return y

    def forward_feature(self, y, base_shape):
        B, channel, K, L = base_shape
        if K == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)
        y = self.feature_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, L, channel, K).permute(0, 2, 3, 1).reshape(B, channel, K * L)
        return y

    def forward(self, x, cond_info, diffusion_emb):

        B, channel, K, L = x.shape



        base_shape = x.shape
        x = x.reshape(B, channel, K * L)

        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(-1)  # (B,channel,1)

        y = x + diffusion_emb

        y = self.forward_time(y, base_shape)
        y = self.forward_feature(y, base_shape)  # (B,channel,K*L)
        y = self.mid_projection(y)  # (B,2*channel,K*L)

        _, cond_dim, _, _ = cond_info.shape
        cond_info = cond_info.reshape(B, cond_dim, K * L)
        cond_info = self.cond_projection(cond_info)  # (B,2*channel,K*L)
        y = y + cond_info

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)  # (B,channel,K*L)
        y = self.output_projection(y)

        residual, skip = torch.chunk(y, 2, dim=1)
        x = x.reshape(base_shape)
        residual = residual.reshape(base_shape)
        skip = skip.reshape(base_shape)
        return (x + residual) / math.sqrt(2.0), skip


class CsdiModel(nn.Module):
    def __init__(self,
                 input_dim=1,
                 feature_dim=2,
                 hidden_dim=64,
                 covariate_dim=0,
                 diffusion_embedding_dim=128,
                 num_steps=50,
                 nheads=8,
                 nlayers=4,
                 emb_time_dim=128,
                 emb_feature_dim=16
                 ):


        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.emb_time_dim = emb_time_dim
        self.emb_feature_dim = emb_feature_dim

        self.input_projection = Conv1d_with_init(input_dim*2, hidden_dim, 1)
        self.output_projection1 = Conv1d_with_init(hidden_dim, hidden_dim, 1)
        self.output_projection2 = Conv1d_with_init(hidden_dim, input_dim, 1)
        nn.init.zeros_(self.output_projection2.weight)

        self.embed_layer = nn.Embedding(
            num_embeddings=self.feature_dim, embedding_dim=self.emb_feature_dim
        )

        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=num_steps,
            embedding_dim=diffusion_embedding_dim,
        )

        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    cond_dim=covariate_dim + input_dim + emb_time_dim + emb_feature_dim,
                    hidden_dim=hidden_dim,
                    diffusion_embedding_dim=diffusion_embedding_dim,
                    nheads=nheads
                )
                for _ in range(nlayers)
            ]
        )

    def time_embedding(self, pos, d_model=128):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(pos.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, d_model, 2).to(pos.device) / d_model
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe
    def set_input_to_diffmodel(self, noisy_data, observed_data, cond_mask):

        cond_obs = cond_mask * observed_data
        noisy_target = (1 - cond_mask) * noisy_data
        total_input = torch.cat([cond_obs, noisy_target], dim=3) # (B,L,K,input_dim*2)
        total_input = total_input.permute(0, 3, 2, 1)  # (B,input_dim*2,K,L)
        return total_input

    def get_side_info(self, cond_mask, side_info):
        B, L, K, input_dim = cond_mask.shape
        observed_tp = side_info[:, :, 0, 0]  # (B,L,1)
        # convert julian time to hours and set the first date to be 0
        observed_tp[:, 1:] = observed_tp[:, 1:] - observed_tp[:, 0:-1]
        observed_tp[:, 0] = 0
        observed_tp = observed_tp * 24
        # round to the nearest hour
        observed_tp = torch.round(observed_tp)


        time_emb = self.time_embedding(observed_tp, self.emb_time_dim)  # (B,L,emb_time_dim)
        time_emb = time_emb.unsqueeze(2)  # (B,L,1,emb_time_dim)
        time_emb = time_emb.expand(B, L, K, self.emb_time_dim)

        feature_emb = self.embed_layer(torch.arange(K).to(cond_mask.device))  # (K,emb_feature_dim)
        feature_emb = feature_emb.unsqueeze(0).unsqueeze(0)  # (1,1,K,emb_feature_dim)
        feature_emb = feature_emb.expand(B, L, K, self.emb_feature_dim) # (B,L,K,emb_feature_dim)

        side_info[:, :, :, 0] = 0  # time is not used as side info
        side_info = side_info.expand(B, L, K, side_info.shape[3])
        cond_info = torch.cat([cond_mask, side_info], dim=3)  # (B,L,K,cond_dim+input_dim)
        cond_info = cond_info.float()
        cond_info = torch.cat([cond_info, time_emb, feature_emb], dim=3)  # (B,L,K,cond_dim+input_dim+emb_time_dim+emb_feature_dim)


        cond_info = cond_info.permute(0, 3, 2, 1)  # (B,cond_dim,K,L)
        return cond_info

    def forward(self, x, mask, noisy_data, diffusion_step, u=None, **kwargs):

        ############################################
        x = x.permute(0, 1, 3, 2)
        mask = mask.permute(0, 1, 3, 2)
        noisy_data = noisy_data.permute(0, 1, 3, 2)
        ############################################

        side_info = u


        hidden_dim = self.hidden_dim

        x = self.set_input_to_diffmodel(noisy_data, x, mask)
        cond_info = self.get_side_info(mask, side_info)

        B, inputdim, K, L = x.shape
        x = x.reshape(B, inputdim, K * L)
        x = x.float()
        x = self.input_projection(x)
        x = F.relu(x)
        x = x.reshape(B, hidden_dim, K, L)


        diffusion_emb = self.diffusion_embedding(diffusion_step)

        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, cond_info, diffusion_emb)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = x.reshape(B, hidden_dim, K * L)
        x = self.output_projection1(x) # (B,hidden_dim,K*L)
        x = F.relu(x)
        x = self.output_projection2(x)  # (B,input_dim,K*L)
        x = x.reshape(B, -1, K, L)  # (B,input_dim,K,L)
        x = x.permute(0, 3, 2, 1)  # (B,L,K,input_dim)

        ############################################
        x = x.permute(0, 1, 3, 2)
        ############################################

        return x

    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument('--covariate_dim', type=int, default=0)
        parser.add_argument('--input_dim', type=int, default=1)
        parser.add_argument('--hidden_dim', type=int, default=64)
        parser.add_argument('--diffusion_embedding_dim', type=int, default=128)
        return parser