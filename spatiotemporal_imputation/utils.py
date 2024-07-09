import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import torch_geometric
from torch_geometric.data import Data
from scipy.spatial import distance_matrix
from torch_geometric.loader import NeighborLoader
from torch_geometric.loader import ClusterData, ClusterLoader
import random
import math
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

class GraphTransformerDataset():
    def __init__(self, y, x, mask, eval_mask, space_coords, time_coords, space_sigma, space_threshold, space_partitions_num, window_size, stride, val_ratio):
        self.y = y.astype(np.float32)
        self.y, self.mean, self.std = self.normalize(self.y, mask, axis='time')
        self.x = x.astype(np.float32) if x is not None else None
        self.space_time_covariate = self.add_additional_space_time_covariate(space_coords, time_coords)
        self.mask = mask.astype(np.float32)
        self.eval_mask = eval_mask.astype(np.float32)
        self.space_coords = space_coords
        self.time_coords = time_coords
        self.space_sigma = space_sigma
        self.space_threshold = space_threshold
        self.space_partitions_num = space_partitions_num
        self.window_size = window_size
        self.stride = stride
        self.val_ratio = val_ratio

        observed_mask = mask - eval_mask
        self.val_mask = observed_mask * (np.random.rand(*mask.shape) < val_ratio).astype(np.float32)
        self.load()

    
    def __len__(self):
        return len(self.y_batch)
    
    def __getitem__(self, idx):
        batch = {}
        batch['y'] = self.y_batch[idx].astype(np.float32)
        if self.x is not None:
            batch['x'] = self.x_batch[idx].astype(np.float32)
        batch['mask'] = self.mask_batch[idx].astype(np.float32)
        batch['eval_mask'] = self.eval_mask_batch[idx].astype(np.float32)
        batch['val_mask'] = self.val_mask_batch[idx].astype(np.float32)
        batch['adj'] = self.adj_batch[idx]
        batch['space_time_covariate'] = self.space_time_covariate_batch[idx].astype(np.float32)
        batch['mean'] = self.mean_batch[idx].astype(np.float32)
        batch['std'] = self.std_batch[idx].astype(np.float32)
        return batch
    


    def normalize(self, y, mask, axis):
        if axis == 'space':
            axis_index = 0
        elif axis == 'time':
            axis_index = 1
        elif axis == 'both':
            # Flatten the array for normalization across all values
            y_flat = y.flatten()
            mask_flat = mask.flatten()
            
            sum_y = np.sum(y_flat * mask_flat)
            count_y = np.sum(mask_flat)
            mean = sum_y / count_y
            
            sum_sq_diff = np.sum(((y_flat - mean) * mask_flat)**2)
            std = np.sqrt(sum_sq_diff / count_y)
            
            if std == 0:
                std = 1e-4
            
            normalized_y = (y_flat - mean) / std
            normalized_y[~mask_flat] = 0
            
            mean_repeated = np.full(y.shape, mean)
            std_repeated = np.full(y.shape, std)
            
            return normalized_y.reshape(y.shape), mean_repeated, std_repeated
        else:
            raise ValueError("axis must be 'space', 'time', or 'both'")
        
        # Compute the mean using sum and mask
        sum_y = np.sum(y * mask, axis=axis_index)
        count_y = np.sum(mask, axis=axis_index)
        mean = sum_y / count_y
        
        # Reshape mean for broadcasting
        mean = np.expand_dims(mean, axis=axis_index)
        
        # Compute the standard deviation using sum and mask
        sum_sq_diff = np.sum(((y - mean) * mask)**2, axis=axis_index)
        std = np.sqrt(sum_sq_diff / count_y)
        
        # Avoid division by zero by setting zero std to 1e-4
        std[std == 0] = 1e-4
        
        # Reshape std for broadcasting
        std = np.expand_dims(std, axis=axis_index)
        
        # Normalize the array along the specified axis
        normalized_y = (y - mean) / std
        
        # Maintain the mask in the normalized array
        normalized_y[~mask] = 0
        
        # Repeat mean and std to match the shape of y
        mean_repeated = np.repeat(mean, y.shape[axis_index], axis=axis_index)
        std_repeated = np.repeat(std, y.shape[axis_index], axis=axis_index)
        
        # Return normalized array and normalization constants
        return normalized_y, mean_repeated, std_repeated


    
    def create_graph(self, space_coords, sigma, epsilon):
        # Compute the pairwise distance matrix
        dist_matrix = distance_matrix(space_coords, space_coords)

        # Create an edge index and weight list
        edge_index = []
        edge_weights = []

        for i in range(dist_matrix.shape[0]):
            for j in range(i+1, dist_matrix.shape[1]):
                weight = np.exp(-dist_matrix[i, j] / (sigma ** 2))
                if weight >= epsilon:
                    edge_index.append([i, j])
                    edge_index.append([j, i])  # Add both directions for undirected graph
                    edge_weights.append(weight)
                    edge_weights.append(weight)

        # Convert edge index and edge weights to tensors
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_weights = torch.tensor(edge_weights, dtype=torch.float)

        # Create graph data object
        node_indices = torch.arange(space_coords.shape[0], dtype=torch.float).view(-1, 1)
        data = Data(x=node_indices, edge_index=edge_index, edge_attr=edge_weights)
        
        return data
    

    def add_additional_space_time_covariate(self, space_coords, time_coords):
        K = len(space_coords)
        L = len(time_coords)
        spatial_covariate = np.tile(space_coords[:, np.newaxis, :], (1, L, 1))
        time_covariate = np.tile(time_coords, (K, 1)).reshape(K, L, 1)
        space_time_covariate = np.concatenate([spatial_covariate, time_covariate], axis=-1)
        return space_time_covariate



    
    def split_into_temporal_batches(self, L, window_size, stride):
        """
        Splits indexes from 1 to L into batches using a sliding window approach.

        Parameters:
        L (int): The total number of indexes.
        window_size (int): The size of each window (batch).
        stride (int): The gap between the start of each window.
        shuffle (bool): Whether to shuffle the list of batches.

        Returns:
        list of lists: A list where each element is a list representing a batch of indexes.
        """

        if (L - window_size) % stride != 0:
            raise ValueError("The combination of L, window_size, and stride does not allow for a perfect tiling of the data.")
            

        batches = []
        for i in range(0, L, stride):
            batch = list(range(i, min(i + window_size, L + 1)))
            batches.append(batch)
            if i + window_size >= L:
                break
        
        return batches

    

    def load(self):

        # partition space
        graph_data = self.create_graph(self.space_coords, self.space_sigma, self.space_threshold)

        if self.space_partitions_num > 1:
        
            cluster_data = ClusterData(graph_data, num_parts=self.space_partitions_num)  
            # 1. Create subgraphs.
            train_loader = ClusterLoader(cluster_data, batch_size=1, shuffle=True)  # 2. Stochastic partioning scheme.

        else:
            train_loader = iter([graph_data])

        y_partitions = []
        if self.x is not None:
            x_partitions = []
        space_time_covariate_partitions = []
        mask_partitions = []
        eval_mask_partitions = []
        val_mask_partitions = []
        edge_index_partitions = []
        edge_weight_partitions = []
        mean_partitions = []
        std_partitions = []

        for step, sub_data in enumerate(train_loader):
            y_partitions.append(self.y[sub_data.x.squeeze().cpu().numpy().astype('int'), :])
            if self.x is not None:
                x_partitions.append(self.x[sub_data.x.squeeze().cpu().numpy().astype('int'), :, :])
            space_time_covariate_partitions.append(self.space_time_covariate[sub_data.x.squeeze().cpu().numpy().astype('int'), :, :])
            mask_partitions.append(self.mask[sub_data.x.squeeze().cpu().numpy().astype('int'), :])
            eval_mask_partitions.append(self.eval_mask[sub_data.x.squeeze().cpu().numpy().astype('int'), :])
            val_mask_partitions.append(self.val_mask[sub_data.x.squeeze().cpu().numpy().astype('int'), :])
            edge_index_partitions.append(sub_data.edge_index)
            edge_weight_partitions.append(sub_data.edge_attr)
            mean_partitions.append(self.mean[sub_data.x.squeeze().cpu().numpy().astype('int'), :])
            std_partitions.append(self.std[sub_data.x.squeeze().cpu().numpy().astype('int'), :])

        self.y_partitions = y_partitions
        if self.x is not None:
            self.x_partitions = x_partitions
    
        
        self.space_time_covariate_partitions = space_time_covariate_partitions
        self.mask_partitions = mask_partitions
        self.eval_mask_partitions = eval_mask_partitions
        self.val_mask_partitions = val_mask_partitions
        self.edge_index_partitions = edge_index_partitions
        self.edge_weight_partitions = edge_weight_partitions
        self.mean_partitions = mean_partitions
        self.std_partitions = std_partitions

        max_y_len = max([len(y) for y in y_partitions])

        # Pad each partition to the maximum size while preserving structure
        y_padded = [np.pad(y, ((0, max_y_len - len(y)), (0, 0)), 'constant', constant_values=0) for y in y_partitions]        
        if self.x is not None:
            x_padded = [np.pad(x, ((0, max_y_len - x.shape[0]), (0, 0), (0, 0)), 'constant', constant_values=0) for x in x_partitions]
        space_time_covariate_padded = [np.pad(stc, ((0, max_y_len - stc.shape[0]), (0, 0), (0, 0)), 'constant', constant_values=0) for stc in space_time_covariate_partitions]
        mask_padded = [np.pad(m, ((0, max_y_len - m.shape[0]), (0, 0)), 'constant', constant_values=0) for m in mask_partitions]
        eval_mask_padded = [np.pad(em, ((0, max_y_len - em.shape[0]), (0, 0)), 'constant', constant_values=0) for em in eval_mask_partitions]
        val_mask_padded = [np.pad(vm, ((0, max_y_len - vm.shape[0]), (0, 0)), 'constant', constant_values=0) for vm in val_mask_partitions]
        mean_padded = [np.pad(m, ((0, max_y_len - m.shape[0]), (0, 0)), 'constant', constant_values=0) for m in mean_partitions]
        std_padded = [np.pad(s, ((0, max_y_len - s.shape[0]), (0, 0)), 'constant', constant_values=1) for s in std_partitions]

        self.y_padded = y_padded
        if self.x is not None:
            self.x_padded = x_padded
        self.space_time_covariate_padded = space_time_covariate_padded
        self.mask_padded = mask_padded
        self.eval_mask_padded = eval_mask_padded
        self.val_mask_padded = val_mask_padded
        self.mean_padded = mean_padded
        self.std_padded = std_padded

        # partition time
        time_partitions = self.split_into_temporal_batches(
            L=len(self.time_coords), 
            window_size=self.window_size,
            stride=self.stride
        )

        self.time_partitions = time_partitions



        y_batch = [i[:, j] for j in time_partitions for i in y_padded]
        if self.x is not None:
            x_batch = [i[:, j, :] for j in time_partitions for i in x_padded]
        space_time_covariate_batch = [i[:, j, :] for j in time_partitions for i in space_time_covariate_padded]
        mask_batch = [i[:, j] for j in time_partitions for i in mask_padded]
        eval_mask_batch = [i[:, j] for j in time_partitions for i in eval_mask_padded]
        val_mask_batch = [i[:, j] for j in time_partitions for i in val_mask_padded]
        mean_batch = [i[:, j] for j in time_partitions for i in mean_padded]
        std_batch = [i[:, j] for j in time_partitions for i in std_padded]

        adj_batch = []
        for i in range(self.space_partitions_num):
            adj = self.edge_index_to_adj(self.edge_index_partitions[i], self.edge_weight_partitions[i], max_y_len)
                    
            deg = torch.sum(adj, dim=-1)  # (B, K)
            deg_inv_sqrt = deg.pow(-0.5).unsqueeze(-1)  # (B, K, 1)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            deg_inv_sqrt_matrix = torch.diag_embed(deg_inv_sqrt.squeeze(-1))  # (B, K, K)
        
            adj = torch.eye(max_y_len) + torch.matmul(deg_inv_sqrt_matrix, torch.matmul(adj, deg_inv_sqrt_matrix))  # (B, K, K)
            adj_batch.append(adj)
        
        self.adj_batch = [i for _ in time_partitions for i in adj_batch]
    

        self.y_batch = y_batch
        if self.x is not None:
            self.x_batch = x_batch
        self.space_time_covariate_batch = space_time_covariate_batch
        self.mask_batch = mask_batch
        self.eval_mask_batch = eval_mask_batch
        self.val_mask_batch = val_mask_batch
        self.mean_batch = mean_batch
        self.std_batch = std_batch


        
            

    def edge_index_to_adj(self, edge_index, edge_weight, num_nodes):
        # edge_index: (2, E)
        # edge_weight: (E)
        # num_nodes: K
        adj = torch.zeros((num_nodes, num_nodes))
        adj[edge_index[0], edge_index[1]] = edge_weight
        return adj


class GraphTransformerDataModule(pl.LightningDataModule):
    def __init__(self, dataset, batch_size=32):
        super(GraphTransformerDataModule, self).__init__()
        self.dataset = dataset
        self.batch_size = batch_size
    
    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
    
    def test_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
    



class DNNDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X.astype(np.float32))
        self.y = torch.tensor(y.astype(np.float32))
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx, :], self.y[idx]



# lightning data module
class DataModule(pl.LightningDataModule):
    def __init__(self, train_dataset, val_dataset, test_dataset, batch_size=32):
        super(DataModule, self).__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)



def interpolate_missing_values(y_st, y_st_missing, mask_st):
    num_nodes, seq_len = y_st.shape
    for k in range(num_nodes):
        for l in range(seq_len):
            y_st_missing[k, :] = pd.Series(y_st_missing[k, :]).interpolate(method='linear', limit_direction='both').values
    y_st_missing[np.isnan(y_st_missing)] = np.nanmean(y_st_missing)

    return y_st_missing

def generate_space_basis_functions(space_coords):
    num_nodes = space_coords.shape[0]

    # spatial basis functions
    # num_basis = [10**2, 19**2, 37**2]
    num_basis = [25, 81, 121]
    knots_1d = [np.linspace(0,1,int(np.sqrt(i))) for i in num_basis]
    # Wendland kernel
    K = 0
    space_basis = np.zeros((num_nodes, sum(num_basis)))
    for res in range(len(num_basis)):
        theta = 1/np.sqrt(num_basis[res])*2.5
        knots_s1, knots_s2 = np.meshgrid(knots_1d[res],knots_1d[res])
        knots = np.column_stack((knots_s1.flatten(),knots_s2.flatten()))
        for i in range(num_basis[res]):
            d = np.linalg.norm(space_coords-knots[i,:],axis=1)/theta
            for j in range(len(d)):
                if d[j] >= 0 and d[j] <= 1:
                    space_basis[j,i + K] = (1-d[j])**6 * (35 * d[j]**2 + 18 * d[j] + 3)/3
                else:
                    space_basis[j,i + K] = 0
        K = K + num_basis[res]
    return space_basis

# def generate_time_basis_functions(time_coords):
#     seq_len = time_coords.shape[0]
#     # time basis functions
#     time_coords = time_coords.reshape(-1, 1)

#     num_basis = [3, 7, 11]
#     knots = [np.linspace(0,1,i) for i in num_basis]
#     # Wendland kernel
#     K = 0 # basis size
#     time_basis = np.zeros((seq_len, sum(num_basis)))
#     for res in range(len(num_basis)):
#         theta = 1/num_basis[res]*2.5
#         for i in range(num_basis[res]):
#             d = np.absolute(time_coords-knots[res][i])/theta
#             for j in range(len(d)):
#                 if d[j] >= 0 and d[j] <= 1:
#                     time_basis[j,i + K] = (1-d[j])**6 * (35 * d[j]**2 + 18 * d[j] + 3)/3
#                 else:
#                     time_basis[j,i + K] = 0
#         K = K + num_basis[res]
#     return time_basis

def generate_time_basis_functions(time_coords):
    seq_len = time_coords.shape[0]

    ## time basis 
    num_basis = [10, 20, 100]
    # std_arr = [0.4, 0.2, 0.1]
    std_arr = [0.3,0.15,0.05]
    mu_knots = [np.linspace(0,1,int(i)) for i in num_basis]

    time_basis = np.zeros((seq_len, sum(num_basis)))
    K = 0
    for res in range(len(num_basis)):
        std = std_arr[res]
        for i in range(num_basis[res]):
            d = np.square(np.absolute(time_coords-mu_knots[res][i]))
            for j in range(len(d)):
                if d[j] >= 0 and d[j] <= 1:
                    time_basis[j,i + K] = np.exp(-0.5 * d[j]/(std**2))
                else:
                    time_basis[j,i + K] = 0
        K = K + num_basis[res]
    return time_basis



def convert_st_data_for_dnn_training(y, mask, eval_mask, x, space_coords, time_coords, additional_st_covariate):
    num_nodes, seq_len = y.shape
    y = y.reshape(-1)
    mask = mask.reshape(-1)
    eval_mask = eval_mask.reshape(-1)

    # scale space coordinates to [0, 1]
    space_coords = (space_coords - space_coords.min(axis=0)) / (space_coords.max(axis=0) - space_coords.min(axis=0))

    # scale time coordinates to [0, 1]
    time_coords = (time_coords - time_coords.min()) / (time_coords.max() - time_coords.min())

    
   
    if additional_st_covariate == 'coord':
        spatial_covariate = space_coords
        time_covariate = time_coords.reshape(-1, 1)
        


    elif additional_st_covariate == 'time_basis':
        time_covariate = generate_time_basis_functions(time_coords)
        spatial_covariate = space_coords

    elif additional_st_covariate == 'space_basis':
        spatial_covariate = generate_space_basis_functions(space_coords)
        time_covariate = time_coords.reshape(-1, 1)

    
    elif additional_st_covariate == 'st_basis':
        time_covariate = generate_time_basis_functions(time_coords)
        spatial_covariate = generate_space_basis_functions(space_coords)

    else:
        raise ValueError('additional_st_covariate should be either coord, time_basis, space_basis, or st_basis')



        

    spatial_covariate_expand = []
    time_covariate_expand = []
    for i in range(num_nodes):
        for j in range(seq_len):
            spatial_covariate_expand.append(spatial_covariate[i, :])
            time_covariate_expand.append(time_covariate[j, :])

    # convert spatial and time covariates to numpy arrays
    spatial_covariate_expand = np.stack(spatial_covariate_expand, axis=0)
    time_covariate_expand = np.stack(time_covariate_expand, axis=0)

    X = np.concatenate([spatial_covariate_expand, time_covariate_expand], axis=1)

    # if there is external covariate
    if x is not None:
        x = x.reshape(num_nodes * seq_len, -1)
        X = np.concatenate([X, x], axis=1)
    
    # split X and y into training and testing sets based on mask
    training_mask = mask * (1 - eval_mask)
    X_train = X[training_mask == 1, :]
    y_train = y[training_mask == 1]
    X_test = X[eval_mask == 1, :]
    y_test = y[eval_mask == 1]

    return X_train, y_train, X_test, y_test





def create_dnn_dataset(st_dataset, additional_st_covariate='coord', val_ratio=0.2, test_ratio=0.1):

    y, x, mask, eval_mask, space_coords, time_coords = st_dataset.y, st_dataset.x, st_dataset.mask, st_dataset.eval_mask, st_dataset.space_coords, st_dataset.time_coords

    if eval_mask is None:
        # create a eval_mask based on test_ratio
        eval_mask = (np.random.rand(*y.shape) < test_ratio).astype(int)
        eval_mask = mask * eval_mask


    X_train, y_train, X_test, y_test = convert_st_data_for_dnn_training(y, mask, eval_mask, x, space_coords, time_coords, additional_st_covariate=additional_st_covariate)

    train_size = int((1 - val_ratio) * len(y_train))
    train_indices = np.random.choice(len(y_train), train_size, replace=False)
    val_indices = np.setdiff1d(np.arange(len(y_train)), train_indices)

    train_dataset = DNNDataset(X_train[train_indices, :], y_train[train_indices])
    val_dataset = DNNDataset(X_train[val_indices, :], y_train[val_indices])
    test_dataset = DNNDataset(X_test, y_test)

    return train_dataset, val_dataset, test_dataset


def create_graph_transformer_dataset(st_dataset, space_sigma, space_threshold, space_partitions_num, window_size, stride, val_ratio):
    y, x, mask, eval_mask, space_coords, time_coords = st_dataset.y, st_dataset.x, st_dataset.mask, st_dataset.eval_mask, st_dataset.space_coords, st_dataset.time_coords

    dataset = GraphTransformerDataset(y, x, mask, eval_mask, space_coords, time_coords, space_sigma, space_threshold, space_partitions_num, window_size, stride, val_ratio)

    return dataset




class CosineSchedulerWithRestarts(LambdaLR):

    def __init__(self, optimizer: Optimizer,
                 num_warmup_steps: int,
                 num_training_steps: int,
                 min_factor: float = 0.1,
                 linear_decay: float = 0.67,
                 num_cycles: int = 1,
                 last_epoch: int = -1):
        """From https://github.com/huggingface/transformers/blob/v4.18.0/src/transformers/optimization.py#L138

        Create a schedule with a learning rate that decreases following the values
        of the cosine function between the initial lr set in the optimizer to 0,
        with several hard restarts, after a warmup period during which it increases
        linearly between 0 and the initial lr set in the optimizer.

        Args:
            optimizer ([`~torch.optim.Optimizer`]):
                The optimizer for which to schedule the learning rate.
            num_warmup_steps (`int`):
                The number of steps for the warmup phase.
            num_training_steps (`int`):
                The total number of training steps.
            num_cycles (`int`, *optional*, defaults to 1):
                The number of hard restarts to use.
            last_epoch (`int`, *optional*, defaults to -1):
                The index of the last epoch when resuming training.
        Return:
            `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
        """

        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                factor = float(current_step) / float(max(1, num_warmup_steps))
                return max(min_factor, factor)
            progress = float(current_step - num_warmup_steps)
            progress /= float(max(1, num_training_steps - num_warmup_steps))
            if progress >= 1.0:
                return 0.0
            factor = (float(num_cycles) * progress) % 1.0
            cos = 0.5 * (1.0 + math.cos(math.pi * factor))
            lin = 1.0 - (progress * linear_decay)
            return max(min_factor, cos * lin)

        super(CosineSchedulerWithRestarts, self).__init__(optimizer, lr_lambda,
                                                          last_epoch)
