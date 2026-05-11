import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # Configuration
    file_path = "./test_predictions.csv"
    
    # Load dataset
    df = pd.read_csv(file_path)

    # Aggregate predictions at the group level.
    # The custom logic mathematically reduces to taking the maximum value:
    # If any item in the group is predicted as 1, the group prediction is 1.
    group_results = df.groupby('group').agg({
        'true_label': 'first',  # Assuming all items in a group share the same true label
        'pred_label': 'max'     # Returns 1 if any pred is 1, else 0
    }).reset_index()

    y_true = group_results['true_label']
    y_pred = group_results['pred_label']

    # Calculate confusion matrix and accuracy
    cm = confusion_matrix(y_true, y_pred)
    accuracy = np.trace(cm) / float(np.sum(cm))
    accuracy_percentage = accuracy * 100

    # Visualize the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='viridis', 
        xticklabels=['0', '1'], 
        yticklabels=['0', '1']
    )
    
    plt.title(f'Group-Level Confusion Matrix\nAccuracy: {accuracy_percentage:.2f}%')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    # Save with high resolution and display
    plt.savefig("group_confusion_matrix.png", dpi=300, bbox_inches='tight')
    print(f"[✓] Confusion matrix saved successfully. Accuracy: {accuracy_percentage:.2f}%")
    plt.show()

if __name__ == "__main__":
    main()