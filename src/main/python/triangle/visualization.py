import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('src/main/python/triangle/dataset.csv', header=None, names=['x', 'y', 'class', 'class_name'])
df['x'] = pd.to_numeric(df['x'], errors='coerce')
df['y'] = pd.to_numeric(df['y'], errors='coerce')
df['class'] = pd.to_numeric(df['class'], errors='coerce')

weights_layer0 = [[-3.262287139892578, -2.8782620429992676], [1.9234658479690552, -1.6480084657669067]]
weights_layer1 = [-3.546369791030884, 3.584458589553833]
biases_layer0 = [0.9883695244789124, 16.352188110351562]
bias_layer1 = [-5.169458389282227]

plt.figure(figsize=(12, 8))

colors = ['blue' if cls == 0 else 'red' for cls in df['class']]
plt.scatter(df['x'], df['y'], c=colors, alpha=0.7, s=30,
           edgecolors='black', linewidth=0.5)

plt.scatter([], [], c='blue', alpha=0.7, s=30, label='Внутри треугольника (0)')
plt.scatter([], [], c='red', alpha=0.7, s=30, label='Снаружи треугольника (1)')

x_min, x_max = df['x'].min() - 0.5, df['x'].max() + 0.5
y_min, y_max = df['y'].min() - 0.5, df['y'].max() + 0.5

x_range = np.linspace(x_min, x_max, 100)

print(x_range)

for neuron_idx in range(2):
    w1 = weights_layer0[0][neuron_idx]
    w2 = weights_layer0[1][neuron_idx]
    b = biases_layer0[neuron_idx]

    if abs(w2) > 1e-6:
        y_line = (-w1 * x_range - b) / w2
        valid_indices = (y_line >= y_min) & (y_line <= y_max)
        if np.any(valid_indices):
            plt.plot(x_range[valid_indices], y_line[valid_indices], linewidth=2, alpha=0.8,
                    label=f'Нейрон скрытого слоя {neuron_idx}')

w1_out = weights_layer1[0]
w2_out = weights_layer1[1]
b_out = bias_layer1[0]

if abs(w2_out) > 1e-6:
    y_line_out = (-w1_out * x_range - b_out) / w2_out
    valid_indices = (y_line_out >= y_min) & (y_line_out <= y_max)
    if np.any(valid_indices):
        plt.plot(x_range[valid_indices], y_line_out[valid_indices], linewidth=3, alpha=0.9, linestyle='--',
                color='black', label='Выходной слой (граница решений)')

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.gca().set_aspect('equal', adjustable='box')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Точки треугольника и линии принятия решений')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('triangle-class.png', dpi=300, bbox_inches='tight')
plt.show()
