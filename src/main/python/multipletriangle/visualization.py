import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('src/main/python/multipletriangle/dataset.csv')

colors = {
    0: 'blue',      # outside_all - синий
    1: 'red',       # triangle_1 - красный
    2: 'green'      # triangle_2 - зеленый
}

plt.figure(figsize=(12, 10))

for class_id in df['class'].unique():
    class_data = df[df['class'] == class_id]
    color = colors.get(class_id, 'gray')
    label = class_data['class_name'].iloc[0] if 'class_name' in class_data.columns else f'Class {class_id}'
    
    plt.scatter(class_data['x'], class_data['y'], 
                c=color, alpha=0.7, s=40,
                edgecolors='black', linewidth=0.3,
                label=label)

plt.xlabel('X координата', fontsize=12)
plt.ylabel('Y координата', fontsize=12)
plt.title('Визуализация датасета с несколькими треугольниками', fontsize=14)

plt.gca().set_aspect('equal', adjustable='box')

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

x_margin = (df['x'].max() - df['x'].min()) * 0.1
y_margin = (df['y'].max() - df['y'].min()) * 0.1

plt.xlim(df['x'].min() - x_margin, df['x'].max() + x_margin)
plt.ylim(df['y'].min() - y_margin, df['y'].max() + y_margin)

plt.tight_layout()
plt.savefig('multiple_triangles_visualization.png', dpi=300, bbox_inches='tight')
plt.show()

print("Статистика по классам:")
print(df['class_name'].value_counts())
print(f"\nОбщее количество точек: {len(df)}")
print(f"Диапазон X: [{df['x'].min():.2f}, {df['x'].max():.2f}]")
print(f"Диапазон Y: [{df['y'].min():.2f}, {df['y'].max():.2f}]")
