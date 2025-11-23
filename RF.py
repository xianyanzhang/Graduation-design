# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib as mpl
import matplotlib.pyplot as plt

# ------------------ SCI Journal Style Settings ------------------
mpl.rcParams.update({
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'lines.linewidth': 1.2,
    'axes.linewidth': 0.8,
    'axes.grid': True,
    'grid.linestyle': '--',
    'grid.linewidth': 0.5,
    'text.usetex': False  # set True if LaTeX is configured
})

# 解决中文显示问题（若需英文，可注释以下两行）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 数据加载与合并
features_df = pd.read_csv("九家湾气象数据.csv", dtype={'时间': str})
target_df   = pd.read_csv("resi指数平均值.csv", dtype={'时间': str})
merged_df = pd.merge(features_df, target_df, on="时间", how="inner")

# 2. 数据预处理
merged_df['平均值'] = merged_df['平均值'].replace(0, np.nan)
merged_df.dropna(subset=['平均值'], inplace=True)
merged_df['时间'] = pd.to_datetime(merged_df['时间'], format='%Y%m')
merged_df.sort_values('时间', inplace=True)
merged_df.reset_index(drop=True, inplace=True)

feature_cols = ['降雨量（mm）', '气温（℃）', '蒸发（mm）', '日照时（h）', '风速（m/s）']
X = merged_df[feature_cols]
y = merged_df['平均值']

# 3. 划分数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

# 4. 模型训练
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

# 5. 评估指标
y_pred = rf_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2  = r2_score(y_test, y_pred)
print(f"MAE: {mae:.4f}, MSE: {mse:.4f}, R²: {r2:.4f}")

# 6. 特征重要性图
importances = pd.Series(rf_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(6, 4))  # 6x4 inches for SCI
importances.plot.bar(ax=ax)
ax.set_ylabel('Importance Score')
ax.set_xlabel('Features')
plt.tight_layout()
plt.savefig('feature_importance.png', bbox_inches='tight')

# 7. 随机样本柱状对比
np.random.seed(42)
sample_idx = np.random.choice(len(y_test), size=min(20, len(y_test)), replace=False)
sample_true = y_test.values[sample_idx]
sample_pred = y_pred[sample_idx]

fig, ax = plt.subplots(figsize=(6, 4))
bar_width = 0.35
idx = np.arange(len(sample_idx))
ax.bar(idx, sample_true, bar_width, label='True')
ax.bar(idx + bar_width, sample_pred, bar_width, label='Pred')
ax.set_xlabel('Sample Index')
ax.set_ylabel('Resi Index')
ax.legend()
plt.tight_layout()
plt.savefig('sample_comparison.png', bbox_inches='tight')

# 8. 散点图：预测 vs 实际
fig, ax = plt.subplots(figsize=(4, 4))
ax.scatter(y_test, y_pred, alpha=0.6)
mn, mx = y.min(), y.max()
ax.plot([mn, mx], [mn, mx], linestyle='--', linewidth=1)
ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
plt.tight_layout()
plt.savefig('pred_vs_actual_scatter.png', bbox_inches='tight')

# 9. 时间序列对比
merged_df['预测值'] = rf_model.predict(merged_df[feature_cols])
fig, ax = plt.subplots(figsize=(6.5, 4))  # two-column width ~6.5 inches
ax.plot(merged_df['时间'], merged_df['平均值'], marker='o', label='Actual')
ax.plot(merged_df['时间'], merged_df['预测值'], marker='x', label='Predicted')
ax.set_xlabel('Time')
ax.set_ylabel('Resi Index')
plt.xticks(rotation=45)
ax.legend()
plt.tight_layout()
plt.savefig('timeseries_comparison.png', bbox_inches='tight')
