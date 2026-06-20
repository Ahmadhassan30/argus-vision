import json

# Modify 04_train_consensus.ipynb
with open('c:/Users/ahmad/Desktop/argus-vision/ml_training/04_train_consensus.ipynb', 'r', encoding='utf-8') as f:
    nb4 = json.load(f)

for cell in nb4['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if "criterion = nn.CrossEntropyLoss(label_smoothing=0.1)" in source:
            new_source = source.replace(
                "criterion = nn.CrossEntropyLoss(label_smoothing=0.1)",
                """from collections import Counter
counts = Counter(ytr.numpy())
total_samples = len(ytr)
class_weights = torch.tensor([
    total_samples / (NUM_CLASSES * max(1.0, counts.get(i, 0.0))) for i in range(NUM_CLASSES)
], dtype=torch.float32).to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)"""
            )
            cell['source'] = [line + ('\n' if i < len(new_source.split('\n')) - 1 and not line.endswith('\n') else '') 
                              for i, line in enumerate(new_source.splitlines(True))]

with open('c:/Users/ahmad/Desktop/argus-vision/ml_training/04_train_consensus.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb4, f, indent=1)


# Modify 05_evaluation.ipynb
with open('c:/Users/ahmad/Desktop/argus-vision/ml_training/05_evaluation.ipynb', 'r', encoding='utf-8') as f:
    nb5 = json.load(f)

for cell in nb5['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if "def metrics_for" in source:
            new_source = source.replace(
                "from sklearn.metrics import balanced_accuracy_score, roc_auc_score",
                "from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score\nfrom sklearn.metrics import confusion_matrix, classification_report"
            )
            new_source = new_source.replace(
                "\"Balanced Accuracy\": float(balanced_accuracy_score(labels, preds)),",
                "\"Accuracy\": float(accuracy_score(labels, preds)),\n        \"Balanced Accuracy\": float(balanced_accuracy_score(labels, preds)),"
            )
            new_source = new_source.replace(
                "for c in [\"Balanced Accuracy\", \"Macro AUC\", \"ECE\"]:",
                "for c in [\"Accuracy\", \"Balanced Accuracy\", \"Macro AUC\", \"ECE\"]:"
            )
            
            # Add confusion matrix print at the end
            if "metrics_table_display\n" in new_source or new_source.strip().endswith("metrics_table_display"):
                new_source = new_source + """

print("\\n=== Argus (full) Classification Report ===")
argus_preds = proba["Argus (full)"].argmax(axis=1)
present_labels = sorted(list(set(y_true) | set(argus_preds)))
print(classification_report(y_true, argus_preds, labels=present_labels, target_names=[ISIC_CLASSES[i] for i in present_labels], zero_division=0))

print("\\n=== Argus (full) Confusion Matrix ===")
cm = confusion_matrix(y_true, argus_preds, labels=list(range(NUM_CLASSES)))
print(pd.DataFrame(cm, index=ISIC_CLASSES, columns=ISIC_CLASSES))
"""
            cell['source'] = [line for line in new_source.splitlines(True)]

with open('c:/Users/ahmad/Desktop/argus-vision/ml_training/05_evaluation.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb5, f, indent=1)
