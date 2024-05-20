import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Ensure the ./figure directory exists
os.makedirs('./figure', exist_ok=True)


############## Figure 1 ##############
# Generate example data with arbitrary movement
np.random.seed(42)  # For reproducibility
n_points = 20
time = range(1, n_points + 1)  # Time from 1 to 20
x = np.cumsum(np.random.randn(n_points))  # Random walk for x
y = np.cumsum(np.random.randn(n_points))  # Random walk for y

df = pd.DataFrame({'time': time, 'x': x, 'y': y})

# Remove observations 10 to 14 to represent missing data
missing_indices = range(9, 14)  # 0-based indexing
# not missing indices
not_missing_indices = [i for i in range(20) if i not in missing_indices]
df_missing = df.drop(missing_indices).reset_index(drop=True)

# Create a new figure
fig = plt.figure(figsize=(10, 5))

# Add the first subfigure (movement in x-y plane)
ax1 = fig.add_subplot(1, 2, 1)
ax1.plot(df_missing.loc[df_missing['time'] <= 9, 'x'], df_missing.loc[df_missing['time'] <= 9, 'y'], marker='x', linestyle='None', color='k')
ax1.plot(df_missing.loc[df_missing['time'] >= 15, 'x'], df_missing.loc[df_missing['time'] >= 15, 'y'], marker='x', linestyle='None', color='k')
ax1.plot(df_missing.loc[df_missing['time'] <= 9, 'x'], df_missing.loc[df_missing['time'] <= 9, 'y'], linestyle='solid', color='green')
ax1.plot(df_missing.loc[df_missing['time'] >= 15, 'x'], df_missing.loc[df_missing['time'] >= 15, 'y'], linestyle='solid', color='green')
ax1.set_title('Movement in x-y Plane')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_xticks([])
ax1.set_yticks([])

# Annotate each point with 1, 2, ..., 20 (excluding missing)
for i, txt in enumerate(df_missing['time']):
    ax1.annotate(txt, (df_missing['x'][i], df_missing['y'][i]), textcoords="offset points", xytext=(5, 5), ha='center')

# Add the second subfigure (x vs. time)
ax2 = fig.add_subplot(2, 2, 2)
ax2.plot(df_missing.loc[df_missing['time'] <= 9, 'time'], df_missing.loc[df_missing['time'] <= 9, 'x'], marker='x', linestyle='None', color='k')
ax2.plot(df_missing.loc[df_missing['time'] >= 15, 'time'], df_missing.loc[df_missing['time'] >= 15, 'x'], marker='x', linestyle='None', color='k')
ax2.plot(df_missing.loc[df_missing['time'] <= 9, 'time'], df_missing.loc[df_missing['time'] <= 9, 'x'], linestyle='solid', color='green')
ax2.plot(df_missing.loc[df_missing['time'] >= 15, 'time'], df_missing.loc[df_missing['time'] >= 15, 'x'], linestyle='solid', color='green')

ax2.set_title('x vs. Time')
ax2.set_xlabel('Time')
ax2.set_ylabel('x')
ax2.tick_params(axis='x', rotation=45)
ax2.set_xticks([])
ax2.set_yticks([])

# Add the third subfigure (y vs. time)
ax3 = fig.add_subplot(2, 2, 4)
ax3.plot(df_missing.loc[df_missing['time'] <= 9, 'time'], df_missing.loc[df_missing['time'] <= 9, 'y'], marker='x', linestyle='None', color='k')
ax3.plot(df_missing.loc[df_missing['time'] >= 15, 'time'], df_missing.loc[df_missing['time'] >= 15, 'y'], marker='x', linestyle='None', color='k')
ax3.plot(df_missing.loc[df_missing['time'] <= 9, 'time'], df_missing.loc[df_missing['time'] <= 9, 'y'], linestyle='solid', color='green')
ax3.plot(df_missing.loc[df_missing['time'] >= 15, 'time'], df_missing.loc[df_missing['time'] >= 15, 'y'], linestyle='solid', color='green')
ax3.set_title('y vs. Time')
ax3.set_xlabel('Time')
ax3.set_ylabel('y')
ax3.tick_params(axis='x', rotation=45)
ax3.set_xticks([])
ax3.set_yticks([])

# Adjust layout to avoid overlap
plt.tight_layout()

# # Show the figure
# plt.show()
# Save the figure
plt.savefig('./figure/algorithm_1.png')
plt.close(fig)

############## Figure 2 ##############

# Replace the missing indices with random noise
df_missing = df.copy()
df_missing.loc[missing_indices, 'x'] = np.random.randn(len(missing_indices)) * 4
df_missing.loc[missing_indices, 'y'] = np.random.randn(len(missing_indices)) * 4

# Create a new figure
fig = plt.figure(figsize=(5, 5))

# Add the second subfigure (x vs. time)
ax1 = fig.add_subplot(2, 1, 1)
ax1.plot(df_missing.loc[not_missing_indices, 'time'], df_missing.loc[not_missing_indices, 'x'], marker='x', linestyle='None', color='k')
ax1.plot(df_missing.loc[missing_indices, 'time'], df_missing.loc[missing_indices, 'x'], marker='o', linestyle='None', color='k', fillstyle='none')
ax1.plot(df_missing['time'], df_missing['x'], linestyle='solid', color='green')
ax1.plot(df_missing.loc[missing_indices, 'time'], df_missing.loc[missing_indices, 'x'], linestyle='solid', color='green')
ax1.set_title('x vs. Time')
ax1.set_xlabel('Time')
ax1.set_ylabel('x')
ax1.tick_params(axis='x', rotation=45)
ax1.set_xticks([])
ax1.set_yticks([])

# Add the third subfigure (y vs. time)
ax2 = fig.add_subplot(2, 1, 2)
ax2.plot(df_missing.loc[not_missing_indices, 'time'], df_missing.loc[not_missing_indices, 'y'], marker='x', linestyle='None', color='k')
ax2.plot(df_missing.loc[missing_indices, 'time'], df_missing.loc[missing_indices, 'y'], marker='o', linestyle='None', color='k', fillstyle='none')
ax2.plot(df_missing['time'], df_missing['y'], linestyle='solid', color='green')
ax2.plot(df_missing.loc[missing_indices, 'time'], df_missing.loc[missing_indices, 'y'], linestyle='solid', color='green')
ax2.set_title('y vs. Time')
ax2.set_xlabel('Time')
ax2.set_ylabel('y')
ax2.tick_params(axis='x', rotation=45)
ax2.set_xticks([])
ax2.set_yticks([])

# Adjust layout to avoid overlap
plt.tight_layout()

# # Show the figure
# plt.show()
# Save the figure
plt.savefig('./figure/algorithm_2.png')
plt.close(fig)

############## Figure 3 ##############
df_missing = df.copy()

# Add some noise to the missing data
df_missing.loc[missing_indices, 'x'] = df_missing.loc[missing_indices, 'x'] + np.random.randn(len(missing_indices))
df_missing.loc[missing_indices, 'y'] = df_missing.loc[missing_indices, 'y'] + np.random.randn(len(missing_indices))

fig = plt.figure(figsize=(5, 5))

# Add the second subfigure (x vs. time)
ax1 = fig.add_subplot(2, 1, 1)
ax1.plot(df_missing.loc[not_missing_indices, 'time'], df_missing.loc[not_missing_indices, 'x'], marker='x', linestyle='None', color='k')
ax1.plot(df_missing.loc[missing_indices, 'time'], df_missing.loc[missing_indices, 'x'], marker='o', linestyle='None', color='k', fillstyle='none')
ax1.plot(df_missing['time'], df_missing['x'], linestyle='solid', color='green')
ax1.plot(df_missing.loc[missing_indices, 'time'], df_missing.loc[missing_indices, 'x'], linestyle='solid', color='green')
ax1.set_title('x vs. Time')
ax1.set_xlabel('Time')
ax1.set_ylabel('x')
ax1.tick_params(axis='x', rotation=45)
ax1.set_xticks([])
ax1.set_yticks([])

# Add the third subfigure (y vs. time)
ax2 = fig.add_subplot(2, 1, 2)
ax2.plot(df_missing.loc[not_missing_indices, 'time'], df_missing.loc[not_missing_indices, 'y'], marker='x', linestyle='None', color='k')
ax2.plot(df_missing.loc[missing_indices, 'time'], df_missing.loc[missing_indices, 'y'], marker='o', linestyle='None', color='k', fillstyle='none')
ax2.plot(df_missing['time'], df_missing['y'], linestyle='solid', color='green')
ax2.plot(df_missing.loc[missing_indices, 'time'], df_missing.loc[missing_indices, 'y'], linestyle='solid', color='green')
ax2.set_title('y vs. Time')
ax2.set_xlabel('Time')
ax2.set_ylabel('y')
ax2.tick_params(axis='x', rotation=45)
ax2.set_xticks([])
ax2.set_yticks([])

# Adjust layout to avoid overlap
plt.tight_layout()

# # Show the figure
# plt.show()
# Save the figure
plt.savefig('./figure/algorithm_3.png')
plt.close(fig)

############## Figure 4 ##############
df_missing = df.copy()

fig = plt.figure(figsize=(5, 5))

# Add the second subfigure (x vs. time)
ax1 = fig.add_subplot(2, 1, 1)
ax1.plot(df_missing.loc[not_missing_indices, 'time'], df_missing.loc[not_missing_indices, 'x'], marker='x', linestyle='None', color='k')
ax1.plot(df_missing.loc[missing_indices, 'time'], df_missing.loc[missing_indices, 'x'], marker='o', linestyle='None', color='k', fillstyle='none')
ax1.plot(df_missing['time'], df_missing['x'], linestyle='solid', color='green')
ax1.plot(df_missing.loc[missing_indices, 'time'], df_missing.loc[missing_indices, 'x'], linestyle='solid', color='green')

ax1.set_title('x vs. Time')
ax1.set_xlabel('Time')
ax1.set_ylabel('x')
ax1.tick_params(axis='x', rotation=45)
ax1.set_xticks([])
ax1.set_yticks([])

# Add the third subfigure (y vs. time)
ax2 = fig.add_subplot(2, 1, 2)
ax2.plot(df_missing.loc[not_missing_indices, 'time'], df_missing.loc[not_missing_indices, 'y'], marker='x', linestyle='None', color='k')
ax2.plot(df_missing.loc[missing_indices, 'time'], df_missing.loc[missing_indices, 'y'], marker='o', linestyle='None', color='k', fillstyle='none')
ax2.plot(df_missing['time'], df_missing['y'], linestyle='solid', color='green')
ax2.plot(df_missing.loc[missing_indices, 'time'], df_missing.loc[missing_indices, 'y'], linestyle='solid', color='green')
ax2.set_title('y vs. Time')
ax2.set_xlabel('Time')
ax2.set_ylabel('y')
ax2.tick_params(axis='x', rotation=45)
ax2.set_xticks([])
ax2.set_yticks([])

# Adjust layout to avoid overlap
plt.tight_layout()

# # Show the figure
# plt.show()
# Save the figure
fig.savefig('./figure/algorithm_4.png')
plt.close(fig)

############## Figure 5 ##############
df_missing = df.copy()
# Create a new figure
fig = plt.figure(figsize=(10, 5))

# Add the first subfigure (movement in x-y plane)
ax1 = fig.add_subplot(1, 2, 1)
ax1.plot(df['x'], df['y'], linestyle='solid', color='green')
ax1.plot(df_missing.loc[missing_indices, 'x'], df_missing.loc[missing_indices, 'y'], marker='x', linestyle='None', color='k')
ax1.plot(df_missing.loc[not_missing_indices, 'x'], df_missing.loc[not_missing_indices, 'y'], marker='o', linestyle='None', color='k', fillstyle='none')
ax1.set_title('Movement in x-y Plane')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_xticks([])
ax1.set_yticks([])

# Annotate each point with 1, 2, ..., 20 (excluding missing)
for i, txt in enumerate(df_missing['time']):
    ax1.annotate(txt, (df_missing['x'][i], df_missing['y'][i]), textcoords="offset points", xytext=(5, 5), ha='center')

# Add the second subfigure (x vs. time)
ax2 = fig.add_subplot(2, 2, 2)
ax2.plot(df_missing.loc[not_missing_indices, 'time'], df_missing.loc[not_missing_indices, 'x'], marker='x', linestyle='None', color='k')
ax2.plot(df_missing.loc[missing_indices, 'time'], df_missing.loc[missing_indices, 'x'], marker='o', linestyle='None', color='k', fillstyle='none')
ax2.plot(df_missing['time'], df_missing['x'], linestyle='solid', color='green')
ax2.plot(df_missing.loc[missing_indices, 'time'], df_missing.loc[missing_indices, 'x'], linestyle='solid', color='green')
ax2.set_title('x vs. Time')
ax2.set_xlabel('Time')
ax2.set_ylabel('x')
ax2.tick_params(axis='x', rotation=45)
ax2.set_xticks([])
ax2.set_yticks([])

# Add the third subfigure (y vs. time)
ax3 = fig.add_subplot(2, 2, 4)
ax3.plot(df_missing.loc[not_missing_indices, 'time'], df_missing.loc[not_missing_indices, 'y'], marker='x', linestyle='None', color='k')
ax3.plot(df_missing.loc[missing_indices, 'time'], df_missing.loc[missing_indices, 'y'], marker='o', linestyle='None', color='k', fillstyle='none')
ax3.plot(df_missing['time'], df_missing['y'], linestyle='solid', color='green')
ax3.plot(df_missing.loc[missing_indices, 'time'], df_missing.loc[missing_indices, 'y'], linestyle='solid', color='green')
ax3.set_title('y vs. Time')
ax3.set_xlabel('Time')
ax3.set_ylabel('y')
ax3.tick_params(axis='x', rotation=45)
ax3.set_xticks([])
ax3.set_yticks([])

# Adjust layout to avoid overlap
plt.tight_layout()

# # Show the figure
# plt.show()

# Save the figure
plt.savefig('./figure/algorithm_5.png')
plt.close(fig)

############## Figure 6 ##############

# Create a new figure
fig = plt.figure(figsize=(5, 5))

# Add the second subfigure (x vs. time)
ax1 = fig.add_subplot(2, 1, 1)
ax1.plot(df_missing.loc[df_missing['time'] <= 9, 'time'], df_missing.loc[df_missing['time'] <= 9, 'x'], marker='x', linestyle='None', color='k')
ax1.plot(df_missing.loc[df_missing['time'] >= 15, 'time'], df_missing.loc[df_missing['time'] >= 15, 'x'], marker='x', linestyle='None', color='k')
ax1.plot(df_missing.loc[df_missing['time'] <= 9, 'time'], df_missing.loc[df_missing['time'] <= 9, 'x'], linestyle='solid', color='green')
ax1.plot(df_missing.loc[df_missing['time'] >= 15, 'time'], df_missing.loc[df_missing['time'] >= 15, 'x'], linestyle='solid', color='green')


ax1.set_title('x vs. Time')
ax1.set_xlabel('Time')
ax1.set_ylabel('x')
ax1.tick_params(axis='x', rotation=45)
ax1.set_xticks([])
ax1.set_yticks([])

# Add the third subfigure (y vs. time)
ax2 = fig.add_subplot(2, 1, 2)
ax2.plot(df_missing.loc[df_missing['time'] <= 9, 'time'], df_missing.loc[df_missing['time'] <= 9, 'y'], marker='x', linestyle='None', color='k')
ax2.plot(df_missing.loc[df_missing['time'] >= 15, 'time'], df_missing.loc[df_missing['time'] >= 15, 'y'], marker='x', linestyle='None', color='k')
ax2.plot(df_missing.loc[df_missing['time'] <= 9, 'time'], df_missing.loc[df_missing['time'] <= 9, 'y'], linestyle='solid', color='green')
ax2.plot(df_missing.loc[df_missing['time'] >= 15, 'time'], df_missing.loc[df_missing['time'] >= 15, 'y'], linestyle='solid', color='green')
ax2.set_title('y vs. Time')
ax2.set_xlabel('Time')
ax2.set_ylabel('y')
ax2.tick_params(axis='x', rotation=45)
ax2.set_xticks([])
ax2.set_yticks([])

# Adjust layout to avoid overlap
plt.tight_layout()

# # Show the figure
# plt.show()

# Save the figure
plt.savefig('./figure/algorithm_6.png')
plt.close(fig)
