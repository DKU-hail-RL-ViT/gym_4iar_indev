import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Moving Average Function
def moving_average(data, window_size):
    """Calculate the moving average with a given window size."""
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

# Load Excel file
file_path = 'FLOPs.xlsx'
data = pd.ExcelFile(file_path)

df = data.parse('Sheet1')

# Data preprocessing
df_melted = df.melt(var_name="Algorithm", value_name="FLOPs")
df_melted['Step'] = df_melted.groupby('Algorithm').cumcount()
df_melted['nmcts'] = df_melted['Algorithm'].str.extract(r'nmcts(\d+)').astype(int)
df_melted['Model Name'] = df_melted['Algorithm'].str.extract(r'([A-Za-z]+)_')

# Convert FLOPs column to numeric and drop NaN
df_melted['FLOPs'] = pd.to_numeric(df_melted['FLOPs'], errors='coerce')
df_melted = df_melted.dropna(subset=['FLOPs'])

# Color map creation
import matplotlib.cm as cm
nmcts_order = [2, 10, 50, 100, 400]
unique_nmcts = sorted(df_melted['nmcts'].unique(), key=lambda x: nmcts_order.index(x))
color_map = dict(zip(unique_nmcts, cm.tab10(np.linspace(0,1,len(unique_nmcts)))))

# Plot settings
plt.figure(figsize=(8, 5))
plt.rcParams.update({'font.size': 16})

# Moving Average Window
window_size = 10

# Line styles based on model
line_styles = {'QRQAC': '--', 'EQRQAC': '-'}

for (model_name, algorithm), group in df_melted.groupby(['Model Name', 'Algorithm']):
    color = color_map[group['nmcts'].iloc[0]]
    line_style = line_styles.get(model_name, '-')
    ma_flops = moving_average(group['FLOPs'], window_size)
    steps = np.arange(len(ma_flops))
    plt.plot(steps, ma_flops, label=f"{model_name}_nmcts{group['nmcts'].iloc[0]}",
             color=color, linestyle=line_style)

# Customize plot
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Axis labels
ax.set_xlabel('Step', fontsize=16)
ax.set_ylabel('FLOPs', fontsize=16)
ax.set_xlabel('Step')
ax.xaxis.set_label_coords(0.98, -0.02)
plt.ylabel('FLOPs')

# Legend adjustment
handles, labels = ax.get_legend_handles_labels()
sorted_legend = sorted(zip(labels, handles), key=lambda x: (
    x[0].startswith('QRQAC'),
    nmcts_order.index(int(x[0].split('nmcts')[-1]))))
sorted_labels, sorted_handles = zip(*sorted_legend)

# Rename QRQAC and EQRQAC in legend
sorted_labels = [label.replace('QRQAC', 'QR-QAC').replace('EQRQAC', 'EQR-QAC') for label in sorted_labels]

plt.legend(sorted_handles, sorted_labels, title='Models', fontsize=16, title_fontsize=16,
           bbox_to_anchor=(0.96, 1),
           loc='upper left', edgecolor='black')

plt.xticks([])
plt.tight_layout()

# Show plot
plt.show()
