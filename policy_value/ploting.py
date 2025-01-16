import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

file_path = "player_elo_result_sh_rldm.csv"
elo_data = pd.read_csv(file_path)

# Check the structure of the data to ensure proper formatting for the plot
elo_data.head(), elo_data.columns
# Extract model names and nmcts values from 'Player Name'
elo_data['Model'] = elo_data['Player Name'].apply(lambda x: re.match(r"([A-Za-z]+)", x).group(0))
elo_data['n_mcts'] = elo_data['Player Name'].apply(
    lambda x: 'efficient' if 'efficient' in x else int(re.search(r"\d+", x).group(0))
)

# Convert n_mcts to numeric for plotting
elo_data = elo_data[elo_data['n_mcts'] != 'efficient']  # 'efficient' 제외
elo_data['n_mcts'] = pd.to_numeric(elo_data['n_mcts'])

# Adjust the font size of all elements as requested
plt.figure(figsize=(7, 5))

# Define larger font size
font_size = 16  # Set a larger font size for axes, labels, and legend

# Set offsets for each model to avoid overlap
model_names = elo_data['Model'].unique()
offsets = np.linspace(-0.2, 0.2, len(model_names))

for idx, model in enumerate(model_names):
    model_data = elo_data[elo_data['Model'] == model]
    # Map n_mcts to categorical indices
    categorical_x = model_data['n_mcts'].apply(lambda x: {2: 0, 10: 1, 50: 2, 100: 3, 400: 4}[x])
    plt.scatter(
        categorical_x + offsets[idx],
        model_data['Elo Rating'],
        label=model,
        alpha=0.7
    )

# Set custom ticks for the x-axis
xticks = [0, 1, 2, 3, 4]
xlabels = [2, 10, 50, 100, 400]
plt.xticks(xticks, labels=xlabels, fontsize=font_size)

# Adjust y-axis font size
plt.yticks(fontsize=font_size)

# Add labels with larger font size
plt.xlabel("Max-depth of planning (n_mcts)", fontsize=font_size)
plt.ylabel("Elo Rating", fontsize=font_size)

# Move legend to the bottom-right and add a black border around the box
plt.legend(
    title="Model",
    loc='lower right',  # Place legend in the bottom-right
    fontsize=font_size,
    title_fontsize=font_size,
    frameon=True,  # Enable the legend box
    edgecolor="black"  # Set black border for legend box
)

# Add grid (only vertical)
plt.grid(axis='y', alpha=0.3)

# Modify axis to display only ㄴ-shaped border
plt.gca().spines['top'].set_visible(False)  # Remove top border
plt.gca().spines['right'].set_visible(False)  # Remove right border

# Ensure proper layout
plt.tight_layout()

# Show the plot
plt.show()
