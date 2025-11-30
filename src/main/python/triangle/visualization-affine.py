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
rotation_angle = 45

transform_matrix_scale = np.array([
    [scale_x, 0],
    [0, scale_y]
])

theta = np.radians(rotation_angle)
rotation_matrix = np.array([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta), np.cos(theta)]
])

points_original = df[['x', 'y']].values
points_scaled = points_original @ transform_matrix_scale
points_rotated = points_original @ rotation_matrix

plt.figure(figsize=(20, 6))

plt.subplot(1, 3, 1)
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

plt.subplot(1, 3, 2)
plt.scatter(points_scaled[:, 0], points_scaled[:, 1], c=colors, alpha=0.7, s=30,
           edgecolors='black', linewidth=0.5)

plt.scatter([], [], c='blue', alpha=0.7, s=30, label='Внутри треугольника (0)')
plt.scatter([], [], c='red', alpha=0.7, s=30, label='Снаружи треугольника (1)')

x_min_s, x_max_s = np.nanmin(points_scaled[:, 0]) - 0.5, np.nanmax(points_scaled[:, 0]) + 0.5
y_min_s, y_max_s = np.nanmin(points_scaled[:, 1]) - 0.5, np.nanmax(points_scaled[:, 1]) + 0.5
x_range_s = np.linspace(x_min_s, x_max_s, 100)

# Преобразуем веса для растяжения
inv_transform_matrix_scale = np.linalg.inv(transform_matrix_scale)
weights_layer0_scaled = inv_transform_matrix_scale.T @ np.array(weights_layer0)

for neuron_idx in range(2):
    w1_s = weights_layer0_scaled[0, neuron_idx]
    w2_s = weights_layer0_scaled[1, neuron_idx]
    b_s = biases_layer0[neuron_idx]

    if abs(w2_s) > 1e-6:
        y_line_s = (-w1_s * x_range_s - b_s) / w2_s
        valid_indices = (y_line_s >= y_min_s) & (y_line_s <= y_max_s)
        valid_indices = valid_indices & ~np.isnan(y_line_s)
        if np.any(valid_indices):
            plt.plot(x_range_s[valid_indices], y_line_s[valid_indices], linewidth=2, alpha=0.8,
                    label=f'Нейрон {neuron_idx} (растяжение)')

plt.xlim(x_min_s, x_max_s)
plt.ylim(y_min_s, y_max_s)
plt.gca().set_aspect('equal', adjustable='box')
plt.xlabel('X (растянуто)')
plt.ylabel('Y (растянуто)')
plt.title(f'После растяжения: X×{scale_x}, Y×{scale_y}')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 3)
plt.scatter(points_rotated[:, 0], points_rotated[:, 1], c=colors, alpha=0.7, s=30,
           edgecolors='black', linewidth=0.5)

plt.scatter([], [], c='blue', alpha=0.7, s=30, label='Внутри треугольника (0)')
plt.scatter([], [], c='red', alpha=0.7, s=30, label='Снаружи треугольника (1)')

x_min_r, x_max_r = np.nanmin(points_rotated[:, 0]) - 0.5, np.nanmax(points_rotated[:, 0]) + 0.5
y_min_r, y_max_r = np.nanmin(points_rotated[:, 1]) - 0.5, np.nanmax(points_rotated[:, 1]) + 0.5
x_range_r = np.linspace(x_min_r, x_max_r, 100)

# Преобразуем веса для поворота
weights_layer0_rotated = np.array(weights_layer0) @ np.array(rotation_matrix)

for neuron_idx in range(2):
    w1_r = weights_layer0_rotated[0, neuron_idx]
    w2_r = weights_layer0_rotated[1, neuron_idx]
    b_r = biases_layer0[neuron_idx]

    if abs(w2_r) > 1e-6:
        y_line_r = (-w1_r * x_range_r - b_r) / w2_r
        valid_indices = (y_line_r >= y_min_r) & (y_line_r <= y_max_r)
        valid_indices = valid_indices & ~np.isnan(y_line_r)
        if np.any(valid_indices):
            plt.plot(x_range_r[valid_indices], y_line_r[valid_indices], linewidth=2, alpha=0.8,
                    label=f'Нейрон {neuron_idx} (поворот)')

plt.xlim(x_min_r, x_max_r)
plt.ylim(y_min_r, y_max_r)
plt.gca().set_aspect('equal', adjustable='box')
plt.xlabel('X (повернуто)')
plt.ylabel('Y (повернуто)')
plt.title(f'После поворота: {rotation_angle}° против часовой')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('triangle-class-affine-rotation.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"Обработано {len(df)} точек данных")
print(f"\nПараметры растяжения: X × {scale_x}, Y × {scale_y}")
print(f"Параметр поворота: {rotation_angle}° против часовой стрелки")

print("\nИсходные веса скрытого слоя:")
print(np.array(weights_layer0))

print("\nВеса после преобразования растяжения:")
print(weights_layer0_scaled)

print("\nВеса после преобразования поворота:")
print(weights_layer0_rotated)

print(f"\nМатрица растяжения:")
print(transform_matrix_scale)

print(f"\nМатрица поворота ({rotation_angle}°):")
print(rotation_matrix)
