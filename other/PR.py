import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.cm as cm

csv_files = {
    "70%-dataset": "./results_70/test_predictions.csv",
    "80%-dataset": "./results_80/test_predictions.csv",
    "90%-dataset": "./results_90/test_predictions.csv",
    "99%-dataset": "./results_99/test_predictions.csv",
}

cmap = cm.get_cmap('viridis', len(csv_files))
colors = [cmap(i) for i in range(len(csv_files))]

plt.figure(figsize=(8, 6))

for i, (model_name, file_path) in enumerate(csv_files.items()):
    df = pd.read_csv(file_path)

    grouped = df.groupby("group").agg(
        true_label=('true_label', lambda x: int((x == 1).any())),
        max_prob=('prob', 'max')
    ).reset_index()

    y_true = grouped['true_label'].values
    y_scores = grouped['max_prob'].values

    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    avg_precision = average_precision_score(y_true, y_scores)

    plt.plot(
        recall,
        precision,
        lw=2,
        color=colors[i],
        label=f'{model_name} (AP = {avg_precision:.4f})'
    )

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision–Recall Curves')
plt.legend(loc='lower left')
plt.grid(False)
plt.tight_layout()
plt.savefig('muti-PR.png', dpi=600)
plt.show()
