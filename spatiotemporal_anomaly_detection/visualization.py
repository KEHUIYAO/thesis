import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

scenario = 'simulation_short_time_series'

data = pd.read_csv(f'./results/{scenario}.csv')



# data manipulation

# if spatial method is 'laws', change it to yes
data.loc[data['spatial method'] == 'laws', 'spatial method'] = 'Yes'

# if spatial method is 'no_laws', change it to no
data.loc[data['spatial method'] == 'no laws', 'spatial method'] = 'No'

# if the temporal method is 'outlier_test', change it to 'Studentized Residual Outlier Test'
data.loc[data['temporal method'] == 'outlier_test', 'temporal method'] = 'Studentized Residual Outlier Test'


# if the temporal method is 'NN', jittered shock magnitude is shock magnitude  - 0.1
data['jittered shock magnitude'] = data['shock magnitude']
data.loc[data['temporal method'] == 'NN', 'jittered shock magnitude'] -= 0.1

# if the temporal method is 'outlier_test', jittered shock magnitude is shock magnitude  + 0.1
data.loc[data['temporal method'] == 'outlier_test', 'jittered shock magnitude'] += 0.1

# capitalize the first letter of the type of anomalies
data['type of anomalies'] = data['type of anomalies'].str.capitalize()

# change trend_seasonal to Trend + Seasonal
data.loc[data['type of time series'] == 'trend_seasonal','type of time series'] = 'Trend + Seasonal'

# change iid_noise to IID Noise
data.loc[data['type of time series'] == 'iid_noise','type of time series'] = 'IID Noise'

# change ar to AR
data.loc[data['type of time series'] == 'ar','type of time series'] = 'AR(2)'

# change the column name temporal method to Time-Series Anomaly Detection Method
data.rename(columns={'temporal method':'Time-Series Anomaly Detection Method'}, inplace=True)

# change the column name spatial method to +LAWS
data.rename(columns={'spatial method':'+LAWS'}, inplace=True)






g = sns.FacetGrid(data, col="type of anomalies", row="type of time series", margin_titles=True, height=3, aspect=1.5)
g.map_dataframe(sns.scatterplot, x="jittered shock magnitude", y="auc", hue="+LAWS", style="Time-Series Anomaly Detection Method", s=100, palette='Set2', markers=['o', 's'], alpha=0.7)
# g.map_dataframe(sns.swarmplot, x="shock magnitude", y="auc", hue="method", palette='Set1', dodge=True)

# add legend on top of the figure
g.add_legend(title='', loc='upper center', bbox_to_anchor=(0.4, 1.2), ncol=1)

g.set_axis_labels("Shock Magnitude", "AUC ")
g.set_titles(col_template="{col_name} Anomaly", row_template="{row_name}")

# Customizing x-axis ticks for each plot
unique_magnitudes = sorted(data['shock magnitude'].unique())
for ax in g.axes.flat:
    ax.set_xticks(unique_magnitudes)
    ax.set_xticklabels(unique_magnitudes)


plt.show()

# save figure to simulation folder as 'simulation_result.png'
g.savefig(f'figure/{scenario}.png', dpi=300)


