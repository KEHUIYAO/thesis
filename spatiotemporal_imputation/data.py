import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import kv, gamma
from scipy.spatial.distance import cdist
from scipy.linalg import cholesky
import os
from sklearn.preprocessing import StandardScaler



class GP():
    def __init__(self, num_nodes, seq_len):
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        self.y, self.space_coords, self.time_coords = self.simulate()
        self.mask = np.where(~np.isnan(self.y), 1, 0)
        self.x = None
        self.eval_mask = None
        

       
    def matern_covariance(self, x1, x2, length_scale=1.0, nu=0.5, sigma=1.0):
        dist = np.linalg.norm(x1 - x2)
        if dist == 0:
            return sigma ** 2
        coeff1 = (2 ** (1 - nu)) / gamma(nu)
        coeff2 = (np.sqrt(2 * nu) * dist) / length_scale
        return sigma ** 2 * coeff1 * (coeff2 ** nu) * kv(nu, coeff2)


    def simulate(self, s_l=1, s_nu=0.5, t_l=5, t_nu=0.5, s_sigma=1, t_sigma=1, seed=42):

        seq_len = self.seq_len
        num_nodes = self.num_nodes

        rng = np.random.RandomState(seed)

        time_coords = np.arange(0, seq_len)
        space_coords = np.random.rand(num_nodes, 2)
        

        # create the temporal covariance matrix
        var_temporal = np.zeros((seq_len, seq_len))
        for i in range(seq_len):
            for j in range(seq_len):
                var_temporal[i, j] = self.matern_covariance(time_coords[i], time_coords[j], t_l, t_nu, t_sigma)

        # create the spatial covariance matrix
        var_spatial = np.zeros((num_nodes, num_nodes))
        for i in range(num_nodes):
            for j in range(num_nodes):
                var_spatial[i, j] = self.matern_covariance(space_coords[i, :], space_coords[j, :], s_l, s_nu, s_sigma)

        L_spatial = cholesky(var_spatial + 1e-6 * np.eye(num_nodes), lower=True)
        L_temporal = cholesky(var_temporal + 1e-6 * np.eye(seq_len), lower=True)

        eta = rng.normal(0, 1, num_nodes * seq_len)
        L_spatial_temporal = np.kron(L_spatial, L_temporal)
        y = np.einsum('ij, j->i', L_spatial_temporal, eta)
        y = y.reshape(num_nodes, seq_len)

        return y, space_coords, time_coords


class KaustCompetition():
    def __init__(self, dataset_index=1):
        self.y, self.eval_mask, self.space_coords, self.time_coords = self.load(dataset_index)
        self.mask = np.where(~np.isnan(self.y), 1, 0)
        self.x = None
    
    def load(self, dataset_index):
        prefix = './raw_data/second_kaust_competition_data/2a/2a_'
        df_train_path = prefix + str(dataset_index) + '_train.csv' 
        df_train = pd.read_csv(df_train_path)
        df_train['eval_mask'] = 0
        X_test_path = prefix + str(dataset_index) + '_test.csv'
        X_test = pd.read_csv(X_test_path)
        y_test_path = './raw_data/second_kaust_competition_data/2a/2a-solutions.csv' 
        y_test = pd.read_csv(y_test_path).loc[:, ['z'+str(dataset_index)]]
        y_test.columns = ['z']

        # concatenate the test data
        df_test = pd.concat([X_test, y_test], axis=1)
        df_test['eval_mask'] = 1

        # concatenate the train and test data
        df = pd.concat([df_train, df_test], axis=0)

        # order by (x,y) and then t
        df = df.sort_values(by=['x', 'y', 't'])

        y = df['z'].values
        eval_mask = df['eval_mask'].values
        space_coords = df[['x', 'y']].drop_duplicates().values
        time_coords = df['t'].drop_duplicates().values
    

        y = y.reshape(len(space_coords), len(time_coords))
        eval_mask = eval_mask.reshape(len(space_coords), len(time_coords))

        return y, eval_mask, space_coords, time_coords



class SoilMoisture():
    def __init__(self):
        date_start = '2016-01-01'
        date_end = '2022-12-31'
        self.y, self.x, self.space_coords, self.time_coords = self.load(date_start, date_end)
        self.mask = np.where(~np.isnan(self.y), 1, 0)
        self.y[self.mask == 0] = 0
        self.eval_mask = self.mask * np.random.choice([0, 1], size=self.y.shape, p=[0.8, 0.2])

    
    def load(self, date_start, date_end):
        df = pd.read_csv('./raw_data/soil_moisture/smap_1km.csv')
        y = df.iloc[:, 4:]


        y = y.T
        tmp = pd.DataFrame(index=pd.date_range(start=date_start, end=date_end))
        y.index = pd.to_datetime(y.index)
        y = tmp.merge(y, left_index=True, right_index=True, how='left')
        y = y.T
        y = y.values


       
        space_coords = df.iloc[:, 2:4].values
        time_coords = np.arange(y.shape[1])

        covariates = ['smap_36km', 'prcp_1km', 'srad_1km', 'tmax_1km', 'tmin_1km', 'vp_1km']
        x_list = []
        for cov in covariates:
            x = pd.read_csv(f'./raw_data/soil_moisture/{cov}.csv')
            x = x.iloc[:, 4:]
            x = x.T
            x.index = pd.to_datetime(x.index)
            tmp = pd.DataFrame(index=pd.date_range(start=date_start, end=date_end))
            x = tmp.merge(x, left_index=True, right_index=True, how='left')
            x = x.T
            x = x.values

            x_mask = ~np.isnan(x)
            x_mask = x_mask.astype(int)
            x[x_mask==0] = np.nanmean(x)

            x_list.append(x)
            x_list.append(x_mask)

        x = np.stack(x_list, axis=-1)

        K, L, C = x.shape
        x = x.reshape((K * L, C))
        scaler = StandardScaler()
        scaler.fit(x)
        x = scaler.transform(x)
        x = x.reshape((K, L, C))

        return y, x, space_coords, time_coords







    

if __name__ == "__main__":
    num_nodes = 3
    seq_len = 4
    gp = GP(num_nodes, seq_len)
    
    
    # generate data with strong temporal correlation and weak spatial correlation
    y_st, y_st_missing, mask_st, space_coords, time_coords = gp.generate_st_data_with_missing_values(t_l=10, s_l=0.1)

    # Plot the data
    plt.figure(figsize=(10, 5))
    plt.imshow(y_st_missing, aspect='auto')
    plt.colorbar()
    plt.title('ST Data with Missing Values')
    plt.xlabel('Time')
    plt.ylabel('Space')
    plt.show()

     # generate data with weak temporal correlation and strong spatial correlation
    y_st, y_st_missing, mask_st, space_coords, time_coords = gp.generate_st_data_with_missing_values(t_l=1, s_l=1.2)

    # Plot the data
    plt.figure(figsize=(10, 5))
    plt.imshow(y_st_missing, aspect='auto')
    plt.colorbar()
    plt.title('ST Data with Missing Values')
    plt.xlabel('Time')
    plt.ylabel('Space')
    plt.show()


    # generate data from the Kaust competition dataset
    kaust = KaustCompetition(dataset_index=1)
    y_st, y_st_missing, mask_st, space_coords, time_coords = kaust.generate_st_data_with_missing_values()
    
    print(y_st_missing.shape)


    
    