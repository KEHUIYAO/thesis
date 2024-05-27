import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from data import AnimalMovement
import os
deer_id_list = [5094]
model_list = ['csdi', 'interpolation', 'crawl']
missing_percent_list = [20, 50, 80]

def generate_non_overlapping_intervals(L, B, offset):
    if B * offset > L:
        raise ValueError(f'B * offset ({B * offset}) must be less than L ({L})')
    starts = []
    end = 0
    for i in range(B):
        # generate integer from end to L - (B-i) * offset
        start = rng.integers(end, L - (B - i) * offset)
        end = start + offset
        starts.append(start)
    return np.array(starts)

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
            offset = 72
            B = 3
            plt.rcParams["font.size"] = 8
            markersize = 2

            # set seed 42
            rng = np.random.default_rng(42)

            starts = generate_non_overlapping_intervals(all_target_np.shape[0], B, offset)

            fig, axes = plt.subplots(nrows=B, ncols=2, figsize=(10, 15))

            # Increase the marker sizes
            circle_markersize = 4  # Size for unfilled circles
            x_markersize = 4 # Size for X marks

            for i, start in enumerate(starts):
                end = start + offset

                for k in range(2):
                    df = pd.DataFrame(
                        {"x": jul[start:end], "val": all_target_np[start:end, k], "y": all_evalpoint_np[start:end, k]})
                    df = df[df.y != 0]
                    df2 = pd.DataFrame(
                        {"x": jul[start:end], "val": all_target_np[start:end, k], "y": all_observed_np[start:end, k]})
                    df2 = df2[df2.y != 0]
                    # Plot green solid line and green shaded area
                    axes[i, k].plot(jul[start:end], quantiles_imp[2][start:end, k], color='g', linestyle='solid',
                                    label='CSDI')
                    axes[i, k].fill_between(jul[start:end], quantiles_imp[0][start:end, k],
                                            quantiles_imp[4][start:end, k],
                                            color='g', alpha=0.3)
                    # Replace red X markers with black X marks
                    axes[i, k].plot(df2.x, df2.val, color='k', marker='x', markersize=x_markersize, linestyle='None')
                    # Replace blue dots with unfilled black circles
                    axes[i, k].plot(df.x, df.val, color='k', marker='o', markersize=circle_markersize, linestyle='None',
                                    fillstyle='none')

                axes[i, 0].set_xlabel('jul')
                axes[i, 1].set_xlabel('jul')
                axes[i, 0].set_ylabel('X')
                axes[i, 1].set_ylabel('Y')
                # Use plain style for y-axis labels to avoid scientific notation
                axes[i, 0].ticklabel_format(style='plain', axis='y')
                axes[i, 1].ticklabel_format(style='plain', axis='y')

            fig.tight_layout(pad=3.0)  # Adjust the pad parameter as needed

            # Save the plot
            plt.savefig(f'./results/{missing_percent}/{deer_id}/{model}/prediction.png', dpi=300)
            plt.savefig(f'./figure/{missing_percent}_{deer_id}_{model}.png', dpi=300)

            plt.close()

######################################### aug #########################################
deer_id = 5094
model = 'csdi'
path = f'./results/aug/{deer_id}/{model}/output.npz'
dataset = AnimalMovement(mode='imputation', deer_id=deer_id)
data = np.load(path)
y_hat = data['y_hat']
y = data['y']
eval_mask = data['eval_mask']
observed_mask = data['observed_mask']
if model == 'csdi':
    imputed_samples = data['imputed_samples']
jul = dataset.attributes['covariates'][:, 0, 0]

# plot the result and save it
all_target_np = y.squeeze(-2)
all_evalpoint_np = eval_mask.squeeze(-2)
all_observed_np = observed_mask.squeeze(-2)
if model == 'csdi':
    samples = imputed_samples.squeeze(-2)
else:
    samples = y_hat.squeeze(-2)[np.newaxis, ...]
qlist = [0.05, 0.25, 0.5, 0.75, 0.95]
quantiles_imp = []
for q in qlist:
    tmp = np.quantile(samples, q, axis=0)
    quantiles_imp.append(tmp * (1 - all_observed_np) + all_target_np * all_observed_np)



offset = 72
B = 3
plt.rcParams["font.size"] = 8
markersize = 2

# set seed 42
rng = np.random.default_rng(42)



# Generate non-overlapping start points


starts = generate_non_overlapping_intervals(all_target_np.shape[0], B, offset)

fig, axes = plt.subplots(nrows=B, ncols=2, figsize=(10, 15))

# Increase the marker sizes
circle_markersize = 4  # Size for the unfilled circles
x_markersize = 4      # Size for the X marks

for i, start in enumerate(starts):
    end = start + offset

    for k in range(2):
        df = pd.DataFrame(
            {"x": jul[start:end], "val": all_target_np[start:end, k], "y": all_evalpoint_np[start:end, k]})
        df = df[df.y != 0]
        df2 = pd.DataFrame(
            {"x": jul[start:end], "val": all_target_np[start:end, k], "y": all_observed_np[start:end, k]})
        df2 = df2[df2.y != 0]
        axes[i, k].plot(jul[start:end], quantiles_imp[2][start:end, k], color='g', linestyle='solid', label='CSDI')
        axes[i, k].fill_between(jul[start:end], quantiles_imp[0][start:end, k], quantiles_imp[4][start:end, k],
                                color='g', alpha=0.3)
        # Replace red X markers with black X marks
        axes[i, k].plot(df2.x, df2.val, color='k', marker='x', markersize=x_markersize, linestyle='None')
        # Replace blue dots with unfilled black circles
        axes[i, k].plot(df.x, df.val, color='k', marker='o', markersize=circle_markersize, linestyle='None', fillstyle='none')

    axes[i, 0].set_xlabel('jul')
    axes[i, 1].set_xlabel('jul')
    axes[i, 0].set_ylabel('X')
    axes[i, 1].set_ylabel('Y')
    # Use plain style for y-axis labels to avoid scientific notation
    axes[i, 0].ticklabel_format(style='plain', axis='y')
    axes[i, 1].ticklabel_format(style='plain', axis='y')

fig.tight_layout(pad=3.0)  # Adjust the pad parameter as needed

# Save the plot
plt.savefig(f'./results/aug/{deer_id}/{model}/prediction.png', dpi=300)
plt.savefig(f'./figure/{deer_id}_4_hour_trajectory_imputation.png', dpi=300)

plt.close()




