import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('src/main/python/triangle/dataset.csv', header=None, names=['x', 'y', 'class', 'class_name'])
df['x'] = pd.to_numeric(df['x'], errors='coerce')
df['y'] = pd.to_numeric(df['y'], errors='coerce')
df['class'] = pd.to_numeric(df['class'], errors='coerce')

weights_layer0 = [[-3.262287139892578, -2.8782620429992676], [1.9234658479690552, -1.6480084657669067]]
biases_layer0 = [0.9883695244789124, 16.352188110351562]

scale_x = 5.0
scale_y = 5.0

transform_matrix = np.array([
    [scale_x, 0],
    [0, scale_y]
])

# Применяем аффинное преобразование к точкам данных
points_original = df[['x', 'y']].values
points_transformed = points_original @ transform_matrix.T

plt.figure(figsize=(15, 6))

# 1. Исходные данные
plt.subplot(1, 2, 1)
colors = ['blue' if cls == 0 else 'red' for cls in df['class']]
plt.scatter(df['x'], df['y'], c=colors, alpha=0.7, s=30,
           edgecolors='black', linewidth=0.5)

plt.scatter([], [], c='blue', alpha=0.7, s=30, label='Внутри треугольника (0)')
plt.scatter([], [], c='red', alpha=0.7, s=30, label='Снаружи треугольника (1)')

x_min, x_max = np.nanmin(df['x']) - 0.5, np.nanmax(df['x']) + 0.5
y_min, y_max = np.nanmin(df['y']) - 0.5, np.nanmax(df['y']) + 0.5
x_range = np.linspace(x_min, x_max, 100)

for neuron_idx in range(2):
    w1 = weights_layer0[0][neuron_idx]
    w2 = weights_layer0[1][neuron_idx]
    b = biases_layer0[neuron_idx]

    if abs(w2) > 1e-6:
        y_line = (-w1 * x_range - b) / w2
        valid_indices = (y_line >= y_min) & (y_line <= y_max)
        valid_indices = valid_indices & ~np.isnan(y_line)
        if np.any(valid_indices):
            plt.plot(x_range[valid_indices], y_line[valid_indices], linewidth=2, alpha=0.8,
                    label=f'Нейрон {neuron_idx}')

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.gca().set_aspect('equal', adjustable='box')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Исходные данные')
plt.legend()
plt.grid(True, alpha=0.3)

# 2. Данные после аффинного преобразования (растяжения)
plt.subplot(1, 2, 2)
plt.scatter(points_transformed[:, 0], points_transformed[:, 1], c=colors, alpha=0.7, s=30,
           edgecolors='black', linewidth=0.5)

plt.scatter([], [], c='blue', alpha=0.7, s=30, label='Внутри треугольника (0)')
plt.scatter([], [], c='red', alpha=0.7, s=30, label='Снаружи треугольника (1)')

x_min_t, x_max_t = np.nanmin(points_transformed[:, 0]) - 0.5, np.nanmax(points_transformed[:, 0]) + 0.5
y_min_t, y_max_t = np.nanmin(points_transformed[:, 1]) - 0.5, np.nanmax(points_transformed[:, 1]) + 0.5
x_range_t = np.linspace(x_min_t, x_max_t, 100)

inv_transform_matrix = np.linalg.inv(transform_matrix)
weights_layer0_transformed = inv_transform_matrix.T @ np.array(weights_layer0)

for neuron_idx in range(2):
    w1_t = weights_layer0_transformed[0, neuron_idx]
    w2_t = weights_layer0_transformed[1, neuron_idx]
    b_t = biases_layer0[neuron_idx]

    if abs(w2_t) > 1e-6:
        y_line_t = (-w1_t * x_range_t - b_t) / w2_t
        valid_indices = (y_line_t >= y_min_t) & (y_line_t <= y_max_t)
        valid_indices = valid_indices & ~np.isnan(y_line_t)
        if np.any(valid_indices):
            plt.plot(x_range_t[valid_indices], y_line_t[valid_indices], linewidth=2, alpha=0.8,
                    label=f'Нейрон {neuron_idx} (преобр.)')

plt.xlim(x_min_t, x_max_t)
plt.ylim(y_min_t, y_max_t)
plt.gca().set_aspect('equal', adjustable='box')
plt.xlabel('X (растянуто)')
plt.ylabel('Y (растянуто)')
plt.title(f'После растяжения: X×{scale_x}, Y×{scale_y}')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('triangle-class-affine.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"Обработано {len(df)} точек данных")
print("Матрица преобразования:")
print(transform_matrix)
print(f"\nПараметры растяжения: X × {scale_x}, Y × {scale_y}")
print("\nИсходные веса скрытого слоя:")
print(np.array(weights_layer0))
print("\nПреобразованные веса скрытого слоя:")
print(weights_layer0_transformed)
