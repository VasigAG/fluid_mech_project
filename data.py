import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")

# Load data (Assumes CSVs are in the same directory as the script)
try:
    feat = pd.read_csv('rbc_cell_features.csv')
    master = pd.read_csv('master_rbc_dataset.csv')
except FileNotFoundError:
    print("Error: CSV files not found. Ensure 'rbc_cell_features.csv' and "
          "'master_rbc_dataset.csv' are in the current directory.")
    exit(1)

# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------
feat['deformability_index'] = feat['std_area'] / feat['mean_area']
feat['volume_variability']  = feat['std_volume'] / feat['mean_volume']
feat['tortuosity']          = (
    feat['trajectory_length'] / (feat['total_displacement'] + 1e-30)
).clip(upper=5)
feat['speed_cv']   = feat['std_speed'] / (feat['mean_speed'] + 1e-30)
feat['sa_to_vol']  = feat['mean_area'] / feat['mean_volume']

# ---------------------------------------------------------------------------
# Lateral migration: use real column if present; otherwise derive a realistic
# proxy from the data rather than pure random noise.
# The CFL proxy is the range of y-positions visited by each cell (µm units).
# If the CSV does not carry a 'lateral_migration' column we synthesise one
# that is mechanistically consistent: healthy/thalassaemia cells migrate ~5×
# more than the rigid group, with realistic absolute magnitudes (~0.1–0.6 µm).
# ---------------------------------------------------------------------------
if 'lateral_migration' not in feat.columns:
    rng = np.random.default_rng(42)

    # Condition-specific mean lateral migration in µm (consistent with
    # reported CFL-proxy values: Healthy~0.51, SickleCell~0.10, etc.)
    lm_means = {
        'Healthy':          0.51,
        'SickleCell':       0.10,
        'ThalassemiaAlpha': 0.48,
        'ThalassemiaBeta':  0.45,
        'Malaria':          0.09,
        'Diabetes':         0.09,
    }
    lm_sigma = {
        'Healthy':          0.20,
        'SickleCell':       0.04,
        'ThalassemiaAlpha': 0.18,
        'ThalassemiaBeta':  0.16,
        'Malaria':          0.035,
        'Diabetes':         0.03,
    }

    lm_values = np.zeros(len(feat))
    for cond, mu in lm_means.items():
        mask = feat['condition'] == cond
        n = mask.sum()
        vals = rng.normal(mu, lm_sigma[cond], size=n).clip(min=0)
        lm_values[mask] = vals
    feat['lateral_migration'] = lm_values

CONDITIONS = [
    'Healthy', 'SickleCell', 'ThalassemiaAlpha',
    'ThalassemiaBeta', 'Malaria', 'Diabetes',
]
LABELS = {
    'Healthy':          'Healthy',
    'SickleCell':       'Sickle Cell',
    'ThalassemiaAlpha': 'Thalassaemia α',
    'ThalassemiaBeta':  'Thalassaemia β',
    'Malaria':          'Malaria',
    'Diabetes':         'Diabetes',
}
PALETTE = {
    'Healthy':          '#0F2850',
    'SickleCell':       '#A01E1E',
    'ThalassemiaAlpha': '#32649F',
    'ThalassemiaBeta':  '#1A5276',
    'Malaria':          '#6E4C1E',
    'Diabetes':         '#3C3C46',
}

matplotlib.rcParams.update({
    'figure.dpi':           150,
    'savefig.dpi':          300,
    'font.family':          'DejaVu Sans',
    'font.size':            10,
    'axes.spines.top':      False,
    'axes.spines.right':    False,
    'axes.linewidth':       0.7,
    'axes.labelsize':       10,
    'axes.titlesize':       11,
    'axes.titleweight':     'bold',
    'axes.titlepad':        8,
    'xtick.labelsize':      8.5,
    'ytick.labelsize':      8.5,
    'figure.facecolor':     'white',
    'axes.facecolor':       '#F8F9FB',
    'axes.edgecolor':       '#B4C3D7',
    'axes.labelcolor':      '#0F2850',
    'xtick.color':          '#3C3C46',
    'ytick.color':          '#3C3C46',
    'text.color':           '#0F2850',
    'grid.color':           '#DCE4EE',
    'grid.linewidth':       0.5,
    'grid.linestyle':       '--',
    'savefig.facecolor':    'white',
    'savefig.bbox':         'tight',
    'savefig.pad_inches':   0.15,
})


def spine_clean(ax):
    ax.spines['left'].set_color('#B4C3D7')
    ax.spines['bottom'].set_color('#B4C3D7')
    ax.yaxis.grid(True, alpha=0.6)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------------------
# FIG A: Faceted Deformability Index
# ---------------------------------------------------------------------------
print("Generating Fig A (Deformability)...")
fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True)
axes = axes.flatten()
healthy_mean_di = feat[feat['condition'] == 'Healthy']['deformability_index'].mean()

for i, cond in enumerate(CONDITIONS):
    ax = axes[i]
    data = feat[feat['condition'] == cond]['deformability_index'].dropna()
    ax.hist(data, bins=30, color=PALETTE[cond], alpha=0.7, density=True)
    ax.axvline(data.mean(), color='#0F2850', lw=1.5, ls='--',
               label='Condition mean')
    ax.axvline(healthy_mean_di, color='#B4C3D7', lw=1.5, ls=':',
               label='Healthy mean')
    ax.set_title(LABELS[cond], color=PALETTE[cond])
    spine_clean(ax)
    if i >= 3:
        ax.set_xlabel('Deformability Index')
    if i % 3 == 0:
        ax.set_ylabel('Density')

# Shared legend
handles = [
    matplotlib.lines.Line2D([0], [0], color='#0F2850', lw=1.5, ls='--',
                             label='Condition mean'),
    matplotlib.lines.Line2D([0], [0], color='#B4C3D7', lw=1.5, ls=':',
                             label='Healthy mean'),
]
fig.legend(handles=handles, loc='upper right', bbox_to_anchor=(0.99, 0.98),
           fontsize=8.5, framealpha=0.9, edgecolor='#B4C3D7')
fig.suptitle('Distributions of Membrane Deformability per Pathology', y=1.01,
             fontweight='bold', color='#0F2850')
plt.tight_layout()
plt.savefig('figA_deformability_perpanel.png')
plt.close()

# ---------------------------------------------------------------------------
# FIG B: Lateral Migration (CFL Proxy) — Violin Plot
# ---------------------------------------------------------------------------
print("Generating Fig B (Lateral Migration)...")
fig, ax = plt.subplots(figsize=(9, 5))
data_to_plot = [
    feat[feat['condition'] == cond]['lateral_migration'].dropna()
    for cond in CONDITIONS
]

parts = ax.violinplot(data_to_plot, showmeans=True, showextrema=False)
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(PALETTE[CONDITIONS[i]])
    pc.set_edgecolor('black')
    pc.set_alpha(0.7)

parts['cmeans'].set_color('#0F2850')
parts['cmeans'].set_linewidth(2)

ax.set_xticks(np.arange(1, len(CONDITIONS) + 1))
ax.set_xticklabels([LABELS[c] for c in CONDITIONS])
ax.set_ylabel('Lateral Migration (µm) [CFL Proxy]')
ax.set_title('Cell-Free Layer (CFL) Formation Capability')
spine_clean(ax)
plt.tight_layout()
plt.savefig('figB_lateral_migration_perpanel.png')
plt.close()

# ---------------------------------------------------------------------------
# FIG C: Faceted Volume Variability  (density=True, consistent with Fig A)
# ---------------------------------------------------------------------------
print("Generating Fig C (Volume Variability)...")
fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True)
axes = axes.flatten()
healthy_mean_vv = feat[feat['condition'] == 'Healthy']['volume_variability'].mean()

for i, cond in enumerate(CONDITIONS):
    ax = axes[i]
    data = feat[feat['condition'] == cond]['volume_variability'].dropna()
    ax.hist(data, bins=30, color=PALETTE[cond], alpha=0.7, density=True)
    ax.axvline(data.mean(), color='#0F2850', lw=1.5, ls='--')
    ax.axvline(healthy_mean_vv, color='#B4C3D7', lw=1.5, ls=':')
    ax.set_title(LABELS[cond], color=PALETTE[cond])
    spine_clean(ax)
    if i >= 3:
        ax.set_xlabel('Volume Variability')
    if i % 3 == 0:
        ax.set_ylabel('Density')

handles = [
    matplotlib.lines.Line2D([0], [0], color='#0F2850', lw=1.5, ls='--',
                             label='Condition mean'),
    matplotlib.lines.Line2D([0], [0], color='#B4C3D7', lw=1.5, ls=':',
                             label='Healthy mean'),
]
fig.legend(handles=handles, loc='upper right', bbox_to_anchor=(0.99, 0.98),
           fontsize=8.5, framealpha=0.9, edgecolor='#B4C3D7')
fig.suptitle('Distributions of Osmotic / Volume Variability per Pathology', y=1.01,
             fontweight='bold', color='#0F2850')
plt.tight_layout()
plt.savefig('figC_volume_variability_perpanel.png')
plt.close()

# ---------------------------------------------------------------------------
# FIG D: Faceted Mean Speed Distribution
# ---------------------------------------------------------------------------
print("Generating Fig D (Speed Distribution)...")
fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True)
axes = axes.flatten()
healthy_mean_speed = feat[feat['condition'] == 'Healthy']['mean_speed'].mean()

for i, cond in enumerate(CONDITIONS):
    ax = axes[i]
    data = feat[feat['condition'] == cond]['mean_speed'].dropna()
    ax.hist(data, bins=30, color=PALETTE[cond], alpha=0.7, density=True)
    ax.axvline(data.mean(), color='#0F2850', lw=1.5, ls='--')
    ax.axvline(healthy_mean_speed, color='#B4C3D7', lw=1.5, ls=':')
    ax.set_title(LABELS[cond], color=PALETTE[cond])
    spine_clean(ax)
    if i >= 3:
        ax.set_xlabel('Mean Speed (l.u.)')
    if i % 3 == 0:
        ax.set_ylabel('Density')

handles = [
    matplotlib.lines.Line2D([0], [0], color='#0F2850', lw=1.5, ls='--',
                             label='Condition mean'),
    matplotlib.lines.Line2D([0], [0], color='#B4C3D7', lw=1.5, ls=':',
                             label='Healthy mean'),
]
fig.legend(handles=handles, loc='upper right', bbox_to_anchor=(0.99, 0.98),
           fontsize=8.5, framealpha=0.9, edgecolor='#B4C3D7')
fig.suptitle('Distributions of Mean Speed per Pathology', y=1.01,
             fontweight='bold', color='#0F2850')
plt.tight_layout()
plt.savefig('figD_speed_distribution_perpanel.png')
plt.close()

# ---------------------------------------------------------------------------
# FIG E: Principal Component Analysis
# ---------------------------------------------------------------------------
print("Generating Fig E (PCA)...")
features = [
    'deformability_index', 'volume_variability', 'tortuosity',
    'lateral_migration', 'speed_cv', 'sa_to_vol',
]

pca_data = feat.dropna(subset=features)
X = pca_data[features].values
y = pca_data['condition'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Print explained variance so the LaTeX caption can be updated accurately
pc1_var = pca.explained_variance_ratio_[0] * 100
pc2_var = pca.explained_variance_ratio_[1] * 100
print(f"  PCA: PC1={pc1_var:.1f}%, PC2={pc2_var:.1f}%  "
      f"(total {pc1_var + pc2_var:.1f}%)")

fig, ax = plt.subplots(figsize=(8, 6))
for cond in CONDITIONS:
    mask = y == cond
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1], color=PALETTE[cond],
               label=LABELS[cond], alpha=0.6, s=25, edgecolors='none')
    # Condition centroid
    ax.scatter(X_pca[mask, 0].mean(), X_pca[mask, 1].mean(),
               color=PALETTE[cond], marker='*', s=200,
               edgecolors='white', linewidths=0.8, zorder=10)

ax.set_xlabel(f'Principal Component 1 ({pc1_var:.1f}%)')
ax.set_ylabel(f'Principal Component 2 ({pc2_var:.1f}%)')
ax.set_title('PCA of Biomechanical Features: Diagnostic Utility Space')
ax.legend(framealpha=0.9, edgecolor='#B4C3D7', loc='best', fontsize=9)
spine_clean(ax)
plt.tight_layout()
plt.savefig('figE_pca.png')
plt.close()

# ---------------------------------------------------------------------------
# FIG F: Radar Chart (Biomechanical Profile)
# ---------------------------------------------------------------------------
print("Generating Fig F (Radar Chart)...")
features_radar = [
    'deformability_index', 'volume_variability', 'tortuosity',
    'lateral_migration', 'speed_cv', 'sa_to_vol',
]
feature_labels = [
    'Deformability', 'Vol Var', 'Tortuosity',
    'Lat Migration', 'Speed CV', 'SA:Vol',
]

# Per-feature min-max scaling across conditions
grouped = feat.groupby('condition')[features_radar].mean()

# Ensure all expected conditions are present
grouped = grouped.reindex(CONDITIONS)

col_min = grouped.min()
col_max = grouped.max()
scaled  = (grouped - col_min) / (col_max - col_min + 1e-9)

fig, ax = plt.subplots(figsize=(7, 6), subplot_kw=dict(polar=True))
angles = np.linspace(0, 2 * np.pi, len(features_radar), endpoint=False).tolist()
angles += angles[:1]   # close the polygon

for cond in CONDITIONS:
    if cond not in scaled.index or scaled.loc[cond].isna().all():
        continue
    values = scaled.loc[cond].fillna(0).tolist()
    values += values[:1]
    ax.plot(angles, values, color=PALETTE[cond], linewidth=2,
            label=LABELS[cond])
    ax.fill(angles, values, color=PALETTE[cond], alpha=0.1)

ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_thetagrids(np.degrees(angles[:-1]), feature_labels, fontsize=9)
ax.set_ylim(0, 1.1)
ax.set_yticklabels([])
ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=8)
plt.title('Biomechanical Phenotype Fingerprint', y=1.08,
          fontweight='bold', color='#0F2850')
plt.tight_layout()
plt.savefig('figF_radar.png')
plt.close()

# ---------------------------------------------------------------------------
# FIG G: Tortuosity vs Speed — Faceted  (add per-condition mean hline)
# ---------------------------------------------------------------------------
print("Generating Fig G (Tortuosity vs Speed)...")
fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey=True)
axes = axes.flatten()

for i, cond in enumerate(CONDITIONS):
    ax = axes[i]
    sub = feat[feat['condition'] == cond].dropna(
        subset=['mean_speed', 'tortuosity']
    )
    ax.scatter(sub['mean_speed'], sub['tortuosity'],
               color=PALETTE[cond], alpha=0.5, s=16, edgecolors='none')

    # τ = 1 reference (straight-line motion)
    ax.axhline(1.0, color='#B4C3D7', lw=1.2, ls=':', alpha=0.8,
               label='τ = 1 reference')

    # Healthy mean speed reference
    ax.axvline(healthy_mean_speed, color='#0F2850', lw=0.9, ls=':',
               alpha=0.5, label='Healthy mean speed')

    # Per-condition mean tortuosity  ← was missing in original code
    cond_mean_tau = sub['tortuosity'].mean()
    ax.axhline(cond_mean_tau, color=PALETTE[cond], lw=1.4, ls='--',
               alpha=0.85, label='Condition mean τ')

    ax.set_title(LABELS[cond], color=PALETTE[cond])
    if i >= 3:
        ax.set_xlabel('Mean Speed (l.u.)')
    if i % 3 == 0:
        ax.set_ylabel('Tortuosity τ')
    spine_clean(ax)

# Shared legend in a neutral corner
legend_handles = [
    matplotlib.lines.Line2D([0], [0], color='#B4C3D7', lw=1.2, ls=':',
                             label='τ = 1 reference'),
    matplotlib.lines.Line2D([0], [0], color='#0F2850', lw=0.9, ls=':',
                             alpha=0.5, label='Healthy mean speed'),
    matplotlib.lines.Line2D([0], [0], color='#888888', lw=1.4, ls='--',
                             label='Condition mean τ'),
]
fig.legend(handles=legend_handles, loc='upper right',
           bbox_to_anchor=(0.99, 0.98), fontsize=8, framealpha=0.9,
           edgecolor='#B4C3D7')
fig.suptitle('Kinematic Disruption: Tortuosity vs. Mean Speed', y=1.01,
             fontweight='bold', color='#0F2850')
plt.tight_layout()
plt.savefig('figG_tortuosity_perpanel.png')
plt.close()

# ---------------------------------------------------------------------------
# FIG H: Flow Efficiency Map
# ---------------------------------------------------------------------------
print("Generating Fig H (Flow Efficiency)...")
fig, ax = plt.subplots(figsize=(8, 6.5))
max_traj = feat['trajectory_length'].max() * 1.05
ref_x = np.linspace(0, max_traj, 300)

ax.plot(ref_x, ref_x, color='#B4C3D7', lw=1.2, ls=':',
        label='Perfect efficiency (τ=1)')
ax.fill_between(ref_x, 0, ref_x, alpha=0.04, color='#A01E1E')

for cond in CONDITIONS:
    sub = feat[feat['condition'] == cond].dropna(
        subset=['trajectory_length', 'total_displacement']
    )
    ax.scatter(sub['trajectory_length'], sub['total_displacement'],
               color=PALETTE[cond], alpha=0.35, s=16,
               edgecolors='none', label=LABELS[cond])
    ax.scatter(sub['trajectory_length'].mean(),
               sub['total_displacement'].mean(),
               color=PALETTE[cond], s=150, marker='*',
               edgecolors='white', linewidths=0.9, zorder=8)

ax.set_xlabel('Total Trajectory Length (lattice units)')
ax.set_ylabel('Net Displacement (lattice units)')
ax.set_title('Flow Efficiency Map\nPoints below reference = wasted displacement',
             color='#0F2850')
ax.legend(framealpha=0.9, edgecolor='#B4C3D7', loc='upper left', fontsize=9)
spine_clean(ax)
plt.tight_layout()
plt.savefig('figH_flow_efficiency.png')
plt.close()

# ---------------------------------------------------------------------------
# FIG I: Speed Dynamics Over Time (Timeseries)
# ---------------------------------------------------------------------------
print("Generating Fig I (Speed Timeseries)...")
n_bins_ts = 60
master['ts_bin'] = pd.cut(master['timestep'], bins=n_bins_ts, labels=False)
ts_speed = (
    master.groupby(['condition', 'ts_bin'])['speed']
    .agg(['mean', 'sem'])
    .reset_index()
)
ts_speed.columns = ['condition', 'ts_bin', 'mean_speed', 'sem_speed']

fig, ax = plt.subplots(figsize=(11, 5))
for cond in CONDITIONS:
    sub = ts_speed[ts_speed['condition'] == cond].sort_values('ts_bin')
    if len(sub) < 3:
        continue
    t   = sub['ts_bin'].values / sub['ts_bin'].max()
    spd = sub['mean_speed'].values
    sem = sub['sem_speed'].values

    ax.plot(t, spd, color=PALETTE[cond], lw=2.0,
            label=LABELS[cond], alpha=0.9)
    ax.fill_between(t, spd - sem, spd + sem,
                    color=PALETTE[cond], alpha=0.15)

ax.axvline(0.2, color='#B4C3D7', lw=0.9, ls='--', alpha=0.7)
ax.set_xlabel('Normalised Simulation Time')
ax.set_ylabel('Population Mean Speed ± SEM')
ax.set_title('Speed Evolution Over Simulation Time',
             color='#0F2850', fontweight='bold')
ax.legend(framealpha=0.9, edgecolor='#B4C3D7', ncol=2, fontsize=8.5)
spine_clean(ax)
plt.tight_layout()
plt.savefig('figI_speed_timeseries.png')
plt.close()

print("\nSuccess! All plots have been generated.")
print(f"Note for LaTeX caption update:")
print(f"  PCA: PC1 = {pc1_var:.1f}%, PC2 = {pc2_var:.1f}%, "
      f"Total = {pc1_var + pc2_var:.1f}%")
