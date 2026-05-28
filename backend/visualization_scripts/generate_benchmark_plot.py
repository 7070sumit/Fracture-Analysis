import matplotlib.pyplot as plt
import os

out_dir = "visualizations/paper_figures"
os.makedirs(out_dir, exist_ok=True)

models = ['Panfilov (Study 2)\nCross-modal Transformer', 'Our Framework\nMultimodal Attention']
aucs = [0.76, 0.845]

plt.figure(figsize=(7, 6))
bars = plt.bar(models, aucs, color=['#7f8c8d', '#27ae60'], width=0.4)

plt.ylim(0.5, 0.95) # Start from 0.5 (random chance) to focus on the improvement
plt.ylabel('ROC-AUC Score', fontsize=12, fontweight='bold')
plt.title('Benchmark Comparison on OAI Dataset', fontsize=14, pad=15, fontweight='bold')

# Add values on top of bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.005, f'{yval:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

out_path = f"{out_dir}/benchmark_comparison.png"
plt.savefig(out_path, dpi=300)
print(f"Generated {out_path}")
