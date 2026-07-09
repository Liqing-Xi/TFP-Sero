import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
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
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.plot(
        fpr,
        tpr,
        lw=2,
        color=colors[i],
        label=f'{model_name} (AUC = {roc_auc:.4f})'
    )

plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('muti-ROC.png', dpi=600)
plt.show()
