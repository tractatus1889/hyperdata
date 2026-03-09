import matplotlib.pyplot as plt
import numpy as np

c1 = '#A8C4E0'
c2 = '#A8D4BC'
c3 = '#F0A8A8'

checkpoints = ['step1000\n(~2B)', 'step36000\n(~72B)', 'step71000\n(~143B)', 'step143000\n(~300B)']
examples_only = [37.2, 33.6, 32.4, 33.4]
meta_1 = [45.9, 35.6, 33.6, 36.9]
meta_10 = [29.9, 28.4, 27.2, 27.0]

x = np.arange(len(checkpoints))
width = 0.3

fig, ax = plt.subplots(figsize=(8, 5))

bars1 = ax.bar(x - width/2, examples_only, width, label='Examples only', color=c1, edgecolor='#333333', linewidth=0.7)
bars2 = ax.bar(x + width/2, meta_1, width, label='Metaexamples 1%', color=c2, edgecolor='#333333', linewidth=0.7)

ax.set_ylabel('Validity rate (%)')
ax.set_xlabel('Pretrain checkpoint')
ax.set_title('Validity rates at checkpoint-3000')
ax.set_xticks(x)
ax.set_xticklabels(checkpoints)
ax.legend()
ax.set_ylim(0, 55)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('/Users/kevinlin/Projects/hyperdata/reports/eval_results.pdf')
plt.savefig('/Users/kevinlin/Projects/hyperdata/reports/eval_results.png', dpi=150)
print('Saved eval_results.pdf and eval_results.png')
