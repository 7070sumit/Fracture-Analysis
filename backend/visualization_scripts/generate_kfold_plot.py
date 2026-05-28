import matplotlib.pyplot as plt
import os
import json

out_dir = "visualizations/paper_figures"
os.makedirs(out_dir, exist_ok=True)

# Data from kfold_results.json
folds = ['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5', 'Average']
aucs = [0.837, 0.827, 0.855, 0.853, 0.852, 0.845]
colors = ['#3498db', '#3498db', '#3498db', '#3498db', '#3498db', '#e74c3c']

plt.figure(figsize=(8, 6))
bars = plt.bar(folds, aucs, color=colors, width=0.5)

plt.ylim(0.75, 0.90) # Zoom in to show the variance clearly
plt.ylabel('ROC-AUC Score', fontsize=12, fontweight='bold')
plt.title('5-Fold Cross Validation Results (Multimodal Attention)', fontsize=14, pad=15, fontweight='bold')

# Add values on top of bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.002, f'{yval:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Add a dashed line for the average
plt.axhline(y=0.845, color='gray', linestyle='--', alpha=0.5)

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

out_path = f"{out_dir}/new_kfold_results.png"
plt.savefig(out_path, dpi=300)
print(f"Generated {out_path}")
