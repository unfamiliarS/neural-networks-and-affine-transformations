import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('src/main/python/multipletriangle/dataset.csv', header=None, names=['x', 'y', 'class', 'class_name'])
df['x'] = pd.to_numeric(df['x'], errors='coerce')
df['y'] = pd.to_numeric(df['y'], errors='coerce')
df['class'] = pd.to_numeric(df['class'], errors='coerce')

weights_layer0 = [
    [-0.3450212776660919, 0.8539338707923889],
    [0.28619298338890076, 0.1603141725063324],
    [-0.09483292698860168, -0.11287638545036316],
    [1.31396484375, -0.980914294719696],
    [0.08046909421682358, -2.3885154724121094],
    [-0.7431932687759399, -0.9621642827987671]
]

biases_layer0 = [-1.1721757650375366, -3.6236443519592285, -2.4966719150543213,
                 -0.05389179289340973, 1.6233192682266235, 5.5311279296875]

plt.figure(figsize=(12, 8))

colors = ['blue' if cls == 0 else 'red' for cls in df['class']]
plt.scatter(df['x'], df['y'], c=colors, alpha=0.7, s=30,
           edgecolors='black', linewidth=0.5)

plt.scatter([], [], c='blue', alpha=0.7, s=30, label='Внутри треугольника (0)')
plt.scatter([], [], c='red', alpha=0.7, s=30, label='Снаружи треугольника (1)')

x_min, x_max = df['x'].min() - 0.5, df['x'].max() + 0.5
y_min, y_max = df['y'].min() - 0.5, df['y'].max() + 0.5

x_range = np.linspace(x_min, x_max, 200)

neuron_colors = ['green', 'orange', 'purple', 'brown', 'pink', 'gray']

for neuron_idx in range(6):
    w1 = weights_layer0[neuron_idx][0]
    w2 = weights_layer0[neuron_idx][1]
    b = biases_layer0[neuron_idx]

    if abs(w2) > 1e-6:
        y_line = (-w1 * x_range - b) / w2
        valid_indices = (y_line >= y_min) & (y_line <= y_max)
        if np.any(valid_indices):
            plt.plot(x_range[valid_indices], y_line[valid_indices], linewidth=2, alpha=0.8,
                    label=f'Нейрон скрытого слоя {neuron_idx}')

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.gca().set_aspect('equal', adjustable='box')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Точки двух треугольников и линии принятия решений (6 нейронов)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('multiple-triangles-class.png', dpi=300, bbox_inches='tight')
plt.show()
