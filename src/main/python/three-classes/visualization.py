import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

weights = [[-11.750592231750488, 11.185347557067871, -0.01008607354015112],
           [-11.846760749816895, 11.143547058105469, 0.01481365505605936]]
biases = [-2.459968090057373, -2.2613184452056885, -2.0144782066345215]

df = pd.read_csv('src/main/python/three-classes/dataset.csv', header=None, names=['x1', 'x2', 'label'])
df['x1'] = pd.to_numeric(df['x1'], errors='coerce')
df['x2'] = pd.to_numeric(df['x2'], errors='coerce')
df['label'] = pd.to_numeric(df['label'], errors='coerce')

plt.figure(figsize=(12, 8))

colors = ['red' if label == 0 else 'blue' if label == 1 else 'green' for label in df['label']]
plt.scatter(df['x1'], df['x2'], c=colors, alpha=0.7, s=30,
           edgecolors='black', linewidth=0.5)
plt.scatter([], [], c='red', alpha=0.7, s=30,
           label='Класс 0', edgecolors='black', linewidth=0.5)
plt.scatter([], [], c='blue', alpha=0.7, s=30,
           label='Класс 1', edgecolors='black', linewidth=0.5)
plt.scatter([], [], c='green', alpha=0.7, s=30,
           label='Класс 2', edgecolors='black', linewidth=0.5)

plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Датасет и линии принятия решений нейронной сети')
plt.grid(True, alpha=0.3)

x1_range = np.linspace(df['x1'].min() - 0.5, df['x1'].max() + 0.5, 100)

for neuron_idx in range(2):
    w1 = weights[0][neuron_idx]
    w2 = weights[1][neuron_idx]
    b = biases[neuron_idx]

    if w2 != 0:
        x2_line = (-w1 * x1_range - b) / w2

        colors = ['purple', 'orange']
        labels = ['Нейрон 0', 'Нейрон 1', ]

        plt.plot(x1_range, x2_line,
                color=colors[neuron_idx],
                linestyle='-',
                linewidth=2.5,
                label=labels[neuron_idx])

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
plt.tight_layout()

plt.savefig('three-class.png', dpi=300, bbox_inches='tight')
plt.show()
