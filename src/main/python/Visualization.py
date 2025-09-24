import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

layer0_weights = [
    [-2.6861188411712646, 2.6660585403442383, -2.018742084503174],
    [-2.936497211456299, 2.4309253692626953, -0.6391273140907288]
]
layer0_biases = [1.8333382606506348, 1.8720186948776245, -0.6206251978874207]

df = pd.read_csv('/home/semyon/projects/neural-network-and-affine-transformations/neural-networks-and-affine-transformations/src/main/python/dataset.csv', header=None, names=['x1', 'x2', 'label'])
df['x1'] = pd.to_numeric(df['x1'], errors='coerce')
df['x2'] = pd.to_numeric(df['x2'], errors='coerce')
df['label'] = pd.to_numeric(df['label'], errors='coerce')

plt.figure(figsize=(12, 8))

# 1. Отображение точек датасета
colors = ['red' if label == 0 else 'blue' for label in df['label']]
plt.scatter(df['x1'], df['x2'], c=colors, alpha=0.7, s=30, 
           label='Класс 0 (красный)', edgecolors='black', linewidth=0.5)
plt.scatter([], [], c='blue', alpha=0.7, s=30, 
           label='Класс 1 (синий)', edgecolors='black', linewidth=0.5)

plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Датасет и линии принятия решений нейронной сети')
plt.grid(True, alpha=0.3)

# 2. Построение линий принятия решений для каждого нейрона
x1_range = np.linspace(df['x1'].min() - 0.5, df['x1'].max() + 0.5, 100)

for neuron_idx in range(3):
    w1 = layer0_weights[0][neuron_idx]
    w2 = layer0_weights[1][neuron_idx]
    b = layer0_biases[neuron_idx]
    
    # Уравнение линии: w1*x1 + w2*x2 + b = 0
    # Выражаем x2 через x1: x2 = (-w1*x1 - b) / w2
    if w2 != 0:  # избегаем деления на ноль
        x2_line = (-w1 * x1_range - b) / w2
        
        colors = ['green', 'orange', 'purple']
        line_styles = ['-', '--', '-.']
        labels = [f'Нейрон 0: -2.69*x1 + 2.67*x2 + 1.83 = 0',
                 f'Нейрон 1: -2.94*x1 + 2.43*x2 + 1.87 = 0',
                 f'Нейрон 2: -2.02*x1 - 0.64*x2 - 0.62 = 0']
        
        plt.plot(x1_range, x2_line, 
                color=colors[neuron_idx], 
                linestyle=line_styles[neuron_idx],
                linewidth=2.5,
                label=labels[neuron_idx])

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
plt.tight_layout()

# Сохранение графика
plt.savefig('dataset_and_decision_lines.png', dpi=300, bbox_inches='tight')
plt.show()
    