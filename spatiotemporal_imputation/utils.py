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

class GraphTransformerDataset():
    def __init__(self, y, x, mask, eval_mask, space_coords, time_coords, space_sigma, space_threshold, space_partitions_num, window_size, stride, val_ratio):
        self.y = y.astype(np.float32)
        self.x = x.astype(np.float32) if x is not None else None
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
        batch['edge_index'] = self.edge_index_batch[idx]
        batch['edge_weight'] = self.edge_weight_batch[idx]
        return batch
    
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
        cluster_data = ClusterData(graph_data, num_parts=self.space_partitions_num)  
        # 1. Create subgraphs.
        train_loader = ClusterLoader(cluster_data, batch_size=1, shuffle=True)  # 2. Stochastic partioning scheme.


        y_partitions = []
        if self.x is not None:
            x_partitions = []
        mask_partitions = []
        eval_mask_partitions = []
        val_mask_partitions = []
        edge_index_partitions = []
        edge_weight_partitions = []

        for step, sub_data in enumerate(train_loader):
            y_partitions.append(self.y[sub_data.x.squeeze().cpu().numpy().astype('int'), :])
            if self.x is not None:
                x_partitions.append(self.x[sub_data.x.squeeze().cpu().numpy().astype('int'), :, :])
            mask_partitions.append(self.mask[sub_data.x.squeeze().cpu().numpy().astype('int'), :])
            eval_mask_partitions.append(self.eval_mask[sub_data.x.squeeze().cpu().numpy().astype('int'), :])
            val_mask_partitions.append(self.val_mask[sub_data.x.squeeze().cpu().numpy().astype('int'), :])
            edge_index_partitions.append(sub_data.edge_index)
            edge_weight_partitions.append(sub_data.edge_attr)

        self.y_partitions = y_partitions
        if self.x is not None:
            self.x_partitions = x_partitions
        self.mask_partitions = mask_partitions
        self.eval_mask_partitions = eval_mask_partitions
        self.val_mask_partitions = val_mask_partitions
        self.edge_index_partitions = edge_index_partitions
        self.edge_weight_partitions = edge_weight_partitions

        max_y_len = max([len(y) for y in y_partitions])
        max_edge_index_len = max([ei.size(1) for ei in edge_index_partitions])
        max_edge_weight_len = max([ew.size(0) for ew in edge_weight_partitions])

        # Pad each partition to the maximum size while preserving structure
        y_padded = [np.pad(y, ((0, max_y_len - len(y)), (0, 0)), 'constant', constant_values=0) for y in y_partitions]        
        if self.x is not None:
            x_padded = [np.pad(x, ((0, max_y_len - x.shape[0]), (0, 0), (0, 0)), 'constant', constant_values=0) for x in x_partitions]
        mask_padded = [np.pad(m, ((0, max_y_len - m.shape[0]), (0, 0)), 'constant', constant_values=0) for m in mask_partitions]
        eval_mask_padded = [np.pad(em, ((0, max_y_len - em.shape[0]), (0, 0)), 'constant', constant_values=0) for em in eval_mask_partitions]
        val_mask_padded = [np.pad(vm, ((0, max_y_len - vm.shape[0]), (0, 0)), 'constant', constant_values=0) for vm in val_mask_partitions]
        edge_index_padded = [np.pad(ei.numpy(), ((0, 0), (0, max_edge_index_len - ei.size(1))), 'constant', constant_values=0) for ei in edge_index_partitions]
        edge_weight_padded = [np.pad(ew.numpy(), (0, max_edge_weight_len - ew.size(0)), 'constant', constant_values=0) for ew in edge_weight_partitions]

        self.y_padded = y_padded
        if self.x is not None:
            self.x_padded = x_padded
        self.mask_padded = mask_padded
        self.eval_mask_padded = eval_mask_padded
        self.val_mask_padded = val_mask_padded
        self.edge_index_padded = edge_index_padded
        self.edge_weight_padded = edge_weight_padded

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
        mask_batch = [i[:, j] for j in time_partitions for i in mask_padded]
        eval_mask_batch = [i[:, j] for j in time_partitions for i in eval_mask_padded]
        val_mask_batch = [i[:, j] for j in time_partitions for i in val_mask_padded]
        edge_index_batch = [i for _ in time_partitions for i in edge_index_padded]
        edge_weight_batch = [i for _ in time_partitions for i in edge_weight_padded]

        self.y_batch = y_batch
        if self.x is not None:
            self.x_batch = x_batch
        self.mask_batch = mask_batch
        self.eval_mask_batch = eval_mask_batch
        self.val_mask_batch = val_mask_batch
        self.edge_index_batch = edge_index_batch
        self.edge_weight_batch = edge_weight_batch


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



