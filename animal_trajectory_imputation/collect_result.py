import os
import re
import pandas as pd
deer_id_list = os.listdir('./results/')

# remove .DS_Store
deer_id_list = [int(x) for x in deer_id_list if x != '.DS_Store']
deer_list = []
mae_interpolation_list = []
mae_transformer_list = []
mae_csdi_list = []
num_eval_points_list = []
crps_list = []

for deer_id in sorted(deer_id_list):
    # print(deer_id)

    # enter the interpolation folder, and read mae.txt file
    try:
        with open(f'./results/{deer_id}/interpolation/mae.txt') as f:
            text = f.read()

            # extract the mae value from 'Test MAE: 123.456\nTest MRE' using regex
            specific_number = re.search(r'Test MAE: (\d+\.\d+|\d+)', text)
            mae_interpolation = float(specific_number.group(1)) if specific_number else None


        # with open(f'./results/{deer_id}/transformer/mae.txt') as f:
        #     mae = f.read()
        #     print(mae)

        with open(f'./results/{deer_id}/csdi/mae.txt') as f:
            text = f.read()
            # extract the mae value from 'Test MAE: 123.456\nTest MRE' using regex
            specific_number = re.search(r'Test MAE: (\d+\.\d+|\d+)', text)
            mae_csdi = float(specific_number.group(1)) if specific_number else None
            num_eval_points = re.search(r'Number of evaluated data points: (\d+)', text)
            num_eval_points = int(num_eval_points.group(1)) if num_eval_points else None

            # extract crps
            crps = re.search(r'CRPS: (\d+\.\d+|\d+)', text)
            crps = float(crps.group(1)) if crps else None






        deer_list.append(deer_id)
        mae_interpolation_list.append(mae_interpolation)
        mae_csdi_list.append(mae_csdi)
        num_eval_points_list.append(num_eval_points)
        crps_list.append(crps)

    except:
        pass


# make a dataframe using deer_list, mae_interpolation_list, mae_csdi_list
df = pd.DataFrame({'deer_id': deer_list, 'mae_interpolation': mae_interpolation_list, 'mae_csdi': mae_csdi_list, 'num_eval_points': num_eval_points_list, 'crps': crps_list})

# count how many times mae_csdi is smaller than mae_interpolation, and average decrease in mae

count = 0
for i in range(len(df)):
    if float(df['mae_csdi'][i]) < float(df['mae_interpolation'][i]):
        count += 1

# print the count
print('CSDI is better than interpolation:', count, 'out of', len(df), 'times')


df['mae_interpolation_total'] = df['mae_interpolation'] * df['num_eval_points']
df['mae_csdi_total'] = df['mae_csdi'] * df['num_eval_points']

df['mae_total_diff']=  df['mae_interpolation_total'] - df['mae_csdi_total']

print('mae for CSDI:', df['mae_csdi_total'].sum() / df['num_eval_points'].sum())
print('mae for interpolation:', df['mae_interpolation_total'].sum() / df['num_eval_points'].sum())
print('Mean of the difference between mae_interpolation and mae_csdi:', df['mae_total_diff'].sum() / df['num_eval_points'].sum())

print('Mean of CRPS:', df['crps'].sum() / len(df))

# save the dataframe to a csv file
df.to_csv('results.csv', index=False)