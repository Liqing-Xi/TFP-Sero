import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, matthews_corrcoef

plt.rcParams['font.family'] = 'Arial'

csv_path = "./test_predictions.csv"

metrics = ['precision', 'recall', 'f1-score', 'accuracy', 'MCC']
labels_short = ['Pr', 'Rc', 'F1', 'Ac', 'MCC']

palette = {
    '0': '#440154',
    '1': '#fde725',
}


df = pd.read_csv(csv_path)

grouped = df.groupby("group").agg(
    true_label=('true_label', lambda x: int((x == 1).any())),
    pred_label=('pred_label', lambda x: int((x == 1).any()))
).reset_index()

y_true = grouped['true_label']
y_pred = grouped['pred_label']
classes = sorted(y_true.unique())

precision, recall, f1, support = precision_recall_fscore_support(
    y_true,
    y_pred,
    labels=classes,
    zero_division=0
)

accuracy = accuracy_score(y_true, y_pred)
mcc = matthews_corrcoef(y_true, y_pred)

plot_data = {}
for i, cls in enumerate(classes):
    plot_data[str(cls)] = [
        precision[i],
        recall[i],
        f1[i],
        accuracy,
        mcc
    ]

angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
angles += angles[:1]  # 闭合

fig, ax = plt.subplots(
    figsize=(6, 6),
    subplot_kw=dict(polar=True)
)

for class_label, values in plot_data.items():
    vals = values + values[:1]
    color = palette.get(class_label)
    ax.plot(
        angles,
        vals,
        linewidth=2,
        label=f'Class {class_label}',
        color=color
    )
    ax.fill(
        angles,
        vals,
        alpha=0.25,
        color=color
    )

ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels_short)
ax.set_ylim(0, 1)
ax.set_yticklabels([])

plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1.15))
plt.tight_layout()

plt.savefig(
    "Radar_chart_99.png",
    dpi=600,
    bbox_inches='tight'
)

plt.show()
