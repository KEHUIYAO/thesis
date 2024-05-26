import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from data import AnimalMovement
import os
deer_id_list = [5094]
model_list = ['csdi']
missing_percent_list = [50]

def generate_non_overlapping_intervals(L, B, offset):
    if B * offset > L:
        raise ValueError(f'B * offset ({B * offset}) must be less than L ({L})')
    starts = []
    end = 0
    for i in range(B):
        start = rng.integer(end, L - (B - i) * offset)
        end = start + offset
        starts.append(start)
    return np.array(starts)

def plot_data_1d(ax, start, end, k):
    # Increase the marker sizes
    circle_markersize = 4  # Size for unfilled circles
    x_markersize = 4  # Size for X marks
    df = pd.DataFrame({"x": jul[start:end], "val": all_target_np[start:end, k], "y": all_evalpoint_np[start:end, k]})
    df = df[df.y != 0]
    df2 = pd.DataFrame({"x": jul[start:end], "val": all_target_np[start:end, k], "y": all_observed_np[start:end, k]})
    df2 = df2[df2.y != 0]
    # Plot green solid line and green shaded area
    ax.plot(jul[start:end], quantiles_imp[2][start:end, k], color='g', linestyle='solid', label='CSDI')
    ax.fill_between(jul[start:end], quantiles_imp[0][start:end, k], quantiles_imp[4][start:end, k], color='g', alpha=0.3)
    # Replace red X markers with black X marks
    ax.plot(df2.x, df2.val, color='k', marker='x', markersize=x_markersize, linestyle='None')
    # Replace blue dots with unfilled black circles
    ax.plot(df.x, df.val, color='k', marker='o', markersize=circle_markersize, linestyle='None', fillstyle='none')
    ax.set_xlabel('jul')
    ax.set_ylabel('X')
    ax.ticklabel_format(style='plain', axis='y')


def plot_data_2d(ax, start, end):
    # Increase the marker sizes
    circle_markersize = 4  # Size for unfilled circles
    x_markersize = 4  # Size for X marks
    df = pd.DataFrame({"t": jul[start:end], "x": all_target_np[start:end, 0], "y": all_target_np[start:end, 1], "eval_mask_x": all_evalpoint_np[start:end, 0], "eval_mask_y": all_evalpoint_np[start:end, 1], 'observed_mask_x': all_observed_np[start:end, 0], 'observed_mask_y': all_observed_np[start:end, 1]})
    df1 = df[df.eval_mask_x != 0]
    df2 = df[df.observed_mask_x != 0]

    # plot 2d trajectory
    ax.plot(df1.x, df1.y, color='k', marker='o', markersize=circle_markersize, linestyle='None', fillstyle='none')
    ax.plot(df2.x, df2.y, color='k', marker='x', markersize=x_markersize, linestyle='None')




# if there is no figure folder, create one
if not os.path.exists('./figure'):
    os.makedirs('./figure')

for deer_id in deer_id_list:
    for model in model_list:
        for missing_percent in missing_percent_list:

            if model == 'crawl':
                path = f'./results/{missing_percent}/{deer_id}/csdi/output.npz'
            else:
                path = f'./results/{missing_percent}/{deer_id}/{model}/output.npz'

            dataset = AnimalMovement(mode='test', deer_id=deer_id)
            data = np.load(path)
            y_hat = data['y_hat']
            y = data['y']
            eval_mask = data['eval_mask']
            observed_mask = data['observed_mask']

            jul = dataset.attributes['covariates'][:, 0, 0]

            if model == 'csdi':
                imputed_samples = data['imputed_samples']
            elif model == 'crawl':
                np.savetxt(f'./crawl/jul.csv', jul, delimiter=',' )
                np.savetxt(f'./crawl/y.csv', y.squeeze(-2), delimiter=',')
                np.savetxt(f'./crawl/eval_mask.csv', eval_mask.squeeze(-2), delimiter=',')

                # run R script to get the imputed samples
                os.system('Rscript ./crawl/visualization.R')
                samples_x = pd.read_csv(f'./crawl/x_pred_samples.csv').values
                samples_y = pd.read_csv(f'./crawl/y_pred_samples.csv').values
                imputed_samples = np.stack([samples_x, samples_y], axis=2)
                imputed_samples = imputed_samples[:, :, np.newaxis, :]

            # plot the result and save it
            all_target_np = y.squeeze(-2)
            all_evalpoint_np = eval_mask.squeeze(-2)
            all_observed_np = observed_mask.squeeze(-2)
            if model == 'csdi' or model == 'crawl':
                samples = imputed_samples.squeeze(-2)
            else:
                samples = y_hat.squeeze(-2)[np.newaxis, ...]

            qlist = [0.05, 0.25, 0.5, 0.75, 0.95]
            quantiles_imp = []
            for q in qlist:
                tmp = np.quantile(samples, q, axis=0)
                quantiles_imp.append(tmp * (1 - all_observed_np) + all_target_np * all_observed_np)


            #######################################
            offset = 20
            B = 1
            plt.rcParams["font.size"] = 8
            markersize = 2

            # set seed 42
            rng = np.random.default_rng(42)







            fig = plt.figure(figsize=(10, 5))

            ax1 = fig.add_subplot(1, 2, 1)
            ax2 = fig.add_subplot(2, 2, 2)
            ax3 = fig.add_subplot(2, 2, 4)

            # plot the trajectory of the deer
            start = generate_non_overlapping_intervals(len(jul), B, offset)[0]
            end = start + offset
            plot_data_1d(ax2, start, end, 0)
            plot_data_1d(ax3, start, end, 1)
            plot_data_2d(ax1, start, end)

            # Save the plot
            plt.savefig(f'./results/{missing_percent}/{deer_id}/{model}/prediction_2d.png', dpi=300)
            plt.savefig(f'./figure/{missing_percent}_{deer_id}_{model}_2d.png', dpi=300)

            plt.close()