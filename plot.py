import re
import math
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv('./gamefile/elo_rating.csv')

base_colors = {
    'AC': sns.color_palette('Reds', 6),         # 빨간색 계열
    'DQN': sns.color_palette('Oranges', 6),     # 주황색 계열
    'EQRQAC': sns.color_palette('YlOrBr', 6),   # 노란색 계열
    'QRAC': sns.color_palette('Blues', 6),      # 파란색 계열
    'QAC': sns.color_palette('Purples', 6),     # 보라색 계열
    'QRDQN': sns.color_palette('Greens', 6),    # 초록색 계열
    'EQRDQN': sns.color_palette('BuGn', 6),     # 청록색 계열
    'QRQAC': sns.color_palette('pink', 6)       # 핑크색 계열
}


pattern = re.compile(
    r'^(?P<model>AC|QRAC|QAC|QRQAC|DQN|QRDQN|EQRDQN|EQRQAC)_'   # 모델 이름들
    r'(?P<cat>nmcts|resource)'                  # nmcts 또는 resource
    r'(?P<number>\d+)'                          # 숫자
    r'(?:_quantiles(?P<quantiles>\d+))?'        # (옵션) quantiles
    r'(?:_eps(?P<eps>[\d\.]+))?$'               # (옵션) eps
)

color_indices = {model: 0 for model in base_colors.keys()}

def get_shaded_color(model_val, total):
    """명도 조절로 색상 밝기 변경"""
    base_palette = base_colors.get(model_val, sns.color_palette('gray', 6))  # 기본 팔레트 적용
    index = color_indices[model_val] % len(base_palette)  # 모델별 색상 인덱스 유지
    color_indices[model_val] += 1  # 다음 색상으로 이동
    return base_palette[index]


def parse_player_name(name: str):
    match = pattern.match(name)
    if match:
        model = match.group('model')
        cat = match.group('cat')  # nmcts 혹은 resource
        number = match.group('number')  # nmcts나 resource 뒤의 숫자
        quantiles = match.group('quantiles')
        eps = match.group('eps')

        # 숫자형 변환
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

parsed_cols = df['Player Name'].apply(parse_player_name)
df_parsed = pd.DataFrame(parsed_cols.tolist())

df = pd.concat([df, df_parsed], axis=1)

forced_nmcts = [2, 10, 50, 100, 400]
df['nmcts_for_plot'] = df['nmcts']
df_resource = df[df['cat'] == 'resource'].copy()
grouped_resource = df_resource.groupby('model', dropna=False)

plt.figure(figsize=(10, 8))

group_cols = ['model', 'cat', 'quantiles', 'eps']
unique_groups = df[group_cols].drop_duplicates().sort_values(by=group_cols)

for _, row in unique_groups.iterrows():
    model_val = row['model']
    cat_val   = row['cat']         # nmcts or resource
    q_val     = row['quantiles']   # None or int
    eps_val   = row['eps']         # None or float

    if model_val in ["AC", "QAC", "DQN", "QRDQN", "QRAC", "QRQAC"]:
        if model_val in ["AC", "QAC"]:
            subset = df[(df['model'] == model_val)].copy()

        elif model_val == "DQN":
            subset = df[
                (df['model'] == model_val) &
                (df['cat'] == cat_val)
                ].copy()

        elif  model_val in ["QRAC", "QRQAC"]:
            subset = df[
                (df['model'] == model_val) &
                (df['cat'] == cat_val) &
                (df['quantiles'] == q_val)
            ].copy()

        elif model_val == "QRDQN":
            subset = df[
                (df['model'] == model_val) &
                (df['cat'] == cat_val) &
                (df['quantiles'] == q_val) &
                (df['eps'] == eps_val)
            ].copy()
    else:
        for model_val, gdf in grouped_resource:
            gdf = gdf.sort_values('resource')

            n = min(len(gdf), 5)
            assigned_nmcts = forced_nmcts[:n]
            idxs = gdf.index[:n]  # 앞에서 n개까지만 매핑

            for i, idx in enumerate(idxs):
                df.loc[idx, 'nmcts_for_plot'] = assigned_nmcts[i]

    subset.sort_values('nmcts', inplace=True)
    x = subset['nmcts']
    y = subset['Elo Rating']

    label_str = f"{model_val}"
    if q_val is not None and not math.isnan(q_val):
        label_str += f"_quantiles{q_val}"
    if eps_val is not None and not math.isnan(eps_val):
        label_str += f"_eps{eps_val}"

    total_lines = len(unique_groups[unique_groups['model'] == model_val])
    color = get_shaded_color(model_val, total_lines)

    if len(subset) == 0:
        continue

    # base_palette = base_colors.get(model_val, 'gray')
    # total_lines = len(unique_groups[unique_groups['model'] == model_val])
    # line_index = len(plt.gca().lines) % total_lines
    #
    # color = get_shaded_color(base_palette, line_index, total_lines)

    plt.plot(x, y, marker='o', label=label_str, color=color)

plt.xlabel("nmcts/resource")

plt.xticks(sorted(df['nmcts'].dropna().unique()))
plt.xlabel("nmcts")
plt.ylabel("Elo Rating")
plt.title("Elo Rating by model / nmcts / quantiles / eps")
plt.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, -0.15),
    ncol=3,
    fontsize='small',
    title='Groups'
)
plt.grid(True)
plt.tight_layout()
plt.show()
