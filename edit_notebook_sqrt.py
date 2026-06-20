import json

# Modify 04_train_consensus.ipynb
with open('c:/Users/ahmad/Desktop/argus-vision/ml_training/04_train_consensus.ipynb', 'r', encoding='utf-8') as f:
    nb4 = json.load(f)

for cell in nb4['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if "from collections import Counter\ncounts = Counter(ytr.numpy())" in source:
            old_str = """from collections import Counter
counts = Counter(ytr.numpy())
total_samples = len(ytr)
class_weights = torch.tensor([
    total_samples / (NUM_CLASSES * max(1.0, counts.get(i, 0.0))) for i in range(NUM_CLASSES)
], dtype=torch.float32).to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)"""
            
            new_str = """from collections import Counter
counts = Counter(ytr.numpy())
total_samples = len(ytr)
raw_weights = [total_samples / (NUM_CLASSES * max(1.0, counts.get(i, 0.0))) for i in range(NUM_CLASSES)]
# Sqrt-inverse-frequency to soften the extremes
sqrt_weights = torch.sqrt(torch.tensor(raw_weights, dtype=torch.float32))
# Cap max weight to 5x minimum weight to handle 1-sample extreme noise
min_w = sqrt_weights.min()
class_weights = torch.clamp(sqrt_weights, max=5.0 * min_w).to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)"""
            
            if old_str in source:
                new_source = source.replace(old_str, new_str)
                cell['source'] = [line + ('\n' if i < len(new_source.split('\n')) - 1 and not line.endswith('\n') else '') 
                                  for i, line in enumerate(new_source.splitlines(True))]

with open('c:/Users/ahmad/Desktop/argus-vision/ml_training/04_train_consensus.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb4, f, indent=1)
