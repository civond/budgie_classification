import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score


csv = "csv/preds_2024_08_06_15_51_28.csv"
df = pd.read_csv(csv)


labels = df['label']
probs = df['probs']
fpr, tpr, thresholds = roc_curve(labels, probs)
roc_auc = auc(fpr, tpr)

num = 831
print(len(thresholds))
print(f"Thresholds: {thresholds[num]}")
print(f"FPR: {fpr[num]}")
print(f"TPR: {tpr[num]}")

df2 = pd.read_csv("csv/preds_2024_07_22_18_01_31.csv")
labels2 = df2['label']
probs2 = df2['probs']
fpr2, tpr2, thresholds = roc_curve(labels2, probs2)
roc_auc2 = auc(fpr2, tpr2)

fig = plt.figure(1, figsize=(5,6))
ax = fig.add_subplot()
plt.plot(fpr, tpr, label=f'Eff. AUC = {roc_auc:0.4f})', color='b')
plt.plot(fpr2, tpr2, label=f'Res. AUC = {roc_auc2:0.4f})', color='r')
plt.plot([0, 1], [0, 1], color='k', alpha=0.3)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')

ax.set_aspect('equal', adjustable='box')

plt.tight_layout()


def convert_csv_to_pred(path, threshold):
    temp_df = pd.read_csv(path)
    temp_df['label'] = (temp_df['probs'] > threshold).astype(int)

    columns_to_keep = ['onset', 'offset', 'label']

    temp_df = temp_df.filter(items=columns_to_keep)
    temp_df = temp_df.sort_values(by='onset', ascending=True)
    temp_df['onset'] = temp_df['onset'].apply(lambda x: f"{x:.6f}")
    temp_df['offset'] = temp_df['offset'].apply(lambda x: f"{x:.6f}")
    temp_df.to_csv("help.txt",header=None, sep='\t', index=False)


    print(temp_df)

convert_csv_to_pred("csv/preds_2024_08_06_15_51_28.csv", 0.2)

#plt.show()