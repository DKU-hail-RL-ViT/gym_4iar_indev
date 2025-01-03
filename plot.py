import re
import math
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pattern = re.compile(
    r'^(?P<model>AC|QRAC|QAC|QRQAC|DQN|QRDQN|EQRDQN|EQRQAC)_'  # model name
    r'(?P<cat>nmcts|resource)'                                # nmcts or resource
    r'(?P<number>\d+)'                                        # number
    r'(?:_quantiles(?P<quantiles>\d+))?'                      # optional quantiles
    r'(?:_eps(?P<eps>[\d\.]+))?$'                             # optional eps
)

# Base color palettes
base_colors = {
    'AC': sns.color_palette('Reds', 1),
    'QRAC': sns.color_palette('Oranges', 4),
    'DQN': sns.color_palette('YlOrBr', 3),
    'QRDQN': sns.color_palette('Greens', 12),
    'QAC': sns.color_palette('BuGn', 1),
    'QRQAC': sns.color_palette('Blues', 4),
    'EQRDQN': sns.color_palette('Purples', 3),
    'EQRQAC': sns.color_palette('RdPu', 1),
}

# Global dictionary to track color indices for each model
color_indices = {model: 0 for model in base_colors.keys()}

def get_shaded_color(model_val):
    """명도를 달리하여 색상을 변경해주는 함수"""
    base_palette = base_colors.get(model_val, sns.color_palette('gray', 6))  # default palette
    index = color_indices[model_val] % len(base_palette)                     # model별로 색상 순서를 유지
    color_indices[model_val] += 1
    return base_palette[index]

def parse_player_name(name: str):
    """주어진 플레이어 이름을 파싱하여 model, cat, nmcts, resource, quantiles, eps 등의 값을 추출"""
    match = pattern.match(name)
    if match:
        model = match.group('model')
        cat = match.group('cat')          # nmcts or resource
        number = match.group('number')    # integer
        quantiles = match.group('quantiles')
        eps = match.group('eps')

        # Convert string to numeric types where applicable
        number = int(number) if number else None
        quantiles = int(quantiles) if quantiles else None
        eps = float(eps) if eps else None

        if cat == 'nmcts':
            nmcts = number
            resource = None
        else:
            nmcts = None
            resource = number

        return {
            'model': model,
            'cat': cat,
            'nmcts': nmcts,
            'resource': resource,
            'quantiles': quantiles,
            'eps': eps
        }
    else:
        return {
            'model': None,
            'cat': None,
            'nmcts': None,
            'resource': None,
            'quantiles': None,
            'eps': None
        }


df = pd.read_csv('./gamefile/elo_rating.csv')

# Parse player name column
parsed_cols = df['Player Name'].apply(parse_player_name)
df_parsed = pd.DataFrame(parsed_cols.tolist())
df = pd.concat([df, df_parsed], axis=1)

# Make a copy for resource category and group by model
df_resource = df[df['cat'] == 'resource'].copy()
grouped_resource = df_resource.groupby('model', dropna=False)

forced_nmcts = [2, 10, 50, 100, 400]
df['nmcts_for_plot'] = df['nmcts']

# Plot settings
plt.figure(figsize=(10, 8))
group_cols = ['model', 'cat', 'quantiles', 'eps']
unique_groups = df[group_cols].drop_duplicates().sort_values(by=group_cols)

# Iterate over each (model, cat, quantiles, eps) group
for _, row in unique_groups.iterrows():
    model_val = row['model']
    cat_val   = row['cat']
    q_val     = row['quantiles']
    eps_val   = row['eps']

    # Filter subset according to the current group
    if model_val in ["AC", "QAC"]:
        subset = df[df['model'] == model_val].copy()

    elif model_val == "DQN":
        subset = df[(df['model'] == model_val) & (df['eps'] == eps_val)].copy()

    elif model_val in ["QRAC", "QRQAC"]:
        subset = df[(df['model'] == model_val) & (df['quantiles'] == q_val)].copy()

    elif model_val == "QRDQN":
        subset = df[(df['model'] == model_val) & (df['quantiles'] == q_val) & (df['eps'] == eps_val)].copy()

    elif model_val == "EQRDQN":
        subset = df[(df['model'] == model_val) & (df['eps'] == eps_val)].copy()
        # Assign forced nmcts
        subset['nmcts'] = forced_nmcts
        n = min(len(subset), len(forced_nmcts))
        assigned_nmcts = forced_nmcts[:n]
        idxs = subset.index[:n]
        for i, idx in enumerate(idxs):
            df.loc[idx, 'nmcts_for_plot'] = assigned_nmcts[i]

    elif model_val == "EQRQAC":
        subset = df[(df['model'] == model_val) & (df['eps'].isnull() | df['eps'].isna())].copy()
        # Assign forced nmcts
        subset['nmcts'] = forced_nmcts
        n = min(len(subset), len(forced_nmcts))
        assigned_nmcts = forced_nmcts[:n]
        idxs = subset.index[:n]
        for i, idx in enumerate(idxs):
            df.loc[idx, 'nmcts_for_plot'] = assigned_nmcts[i]

    subset.sort_values('nmcts', inplace=True)

    x = subset['nmcts']
    y = subset['Elo Rating']

    # Label string
    label_str = f"{model_val}"
    if q_val is not None and not math.isnan(q_val):
        label_str += f"_quantiles{q_val}"
    if eps_val is not None and not math.isnan(eps_val):
        label_str += f"_eps{eps_val}"

    # Get color
    total_lines = len(unique_groups[unique_groups['model'] == model_val])
    color = get_shaded_color(model_val)

    if len(subset) > 0:
        plt.plot(x, y, marker='o', label=label_str, color=color)

# Set plot labels and title
plt.xlabel("nmcts")
plt.ylabel("Elo Rating")
plt.title("Elo Rating by model / nmcts / quantiles / eps")
plt.xticks(sorted(df['nmcts'].dropna().unique()))

plt.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, -0.15),
    ncol=3,
    fontsize='small',
    title='Legends'
)
plt.grid(True)
plt.tight_layout()
plt.show()
