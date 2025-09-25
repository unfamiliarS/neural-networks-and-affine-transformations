import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

layer1_weights = [8.604686737060547, 8.985509872436523]
layer1_biases = 0.05630039796233177

df = pd.read_csv('src/main/python/one-class/dataset.csv', header=None, names=['x1', 'x2', 'label'])
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

w1 = layer1_weights[0]
w2 = layer1_weights[1]
b = layer1_biases

# Уравнение линии: w1*x1 + w2*x2 + b = 0
# Выражаем x2 через x1: x2 = (-w1*x1 - b) / w2
if w2 != 0:
    x2_line = (-w1 * x1_range - b) / w2
    
    plt.plot(x1_range, x2_line, 
            color='green', 
            linestyle='-',
            linewidth=2.5,
            label='Нейрон 0')

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
plt.tight_layout()

plt.savefig('one-class.png', dpi=300, bbox_inches='tight')
plt.show()
    