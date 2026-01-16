import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import argparse

def parse_weights_biases(weights_str, biases_str):
    weights = eval(weights_str)
    biases = eval(biases_str)
    return weights, biases

def apply_rotation(data, angle_degrees):
    angle_rad = np.radians(angle_degrees)
    rotation_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad), np.cos(angle_rad)]
    ])
    result = rotation_matrix @ data.T
    return result.T

def apply_scale(data, scale_x, scale_y, isData=True):
    if isData:
        scale_matrix = np.array([
            [scale_x, 0],
            [0, scale_y]
        ])
    else:
        scale_matrix = np.array([
            [1/scale_x, 0],
            [0, 1/scale_y]
        ])
    result = scale_matrix @ data.T
    return result.T

def apply_shear(data, shear_x, shear_y, isData=True):
    if isData:
        shear_matrix = np.array([
            [1, shear_x],
            [shear_y, 1]
        ])
    else:
        det = 1 - shear_x * shear_y
        shear_matrix = np.array([
            [1, -shear_y],
            [-shear_x, 1]
        ]) / det
    result = shear_matrix @ data.T
    return result.T

def safe_limits(data):
    clean_data = data[np.isfinite(data)]
    if len(clean_data) == 0:
        return -1, 1
    data_min = np.min(clean_data)
    data_max = np.max(clean_data)
    return data_min, data_max

def plot_decision_boundary(ax, weights, biases, x_lim, y_lim, alpha=0.8, linewidth=2):
    x_min, x_max = x_lim
    y_min, y_max = y_lim

    x_padding = (x_max - x_min) * 0.5
    x_range_ext = np.linspace(x_min - x_padding, x_max + x_padding, 400)
    x_range = np.linspace(x_min, x_max, 400)

    colors = ['red', 'blue', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta']

    for neuron_idx in range(len(weights)):
        w1 = weights[neuron_idx][0]
        w2 = weights[neuron_idx][1]
        b = biases[neuron_idx]

        line_color = colors[neuron_idx % len(colors)]
        label = f'Neuron line {neuron_idx}: w=[{w1:.3f}, {w2:.3f}], b={b:.3f}'

        y_line_ext = (-w1 * x_range_ext - b) / w2
        y_line = (-w1 * x_range - b) / w2
        ax.plot(x_range_ext, y_line_ext,
                color=line_color, linewidth=linewidth, alpha=alpha, label=label)

        valid_indices = (y_line >= y_min) & (y_line <= y_max)
        if np.any(valid_indices):
            x_vals = x_range[valid_indices]
            y_vals = y_line[valid_indices]

            idx_mid = len(x_vals) // 2
            x_mid = x_vals[idx_mid]
            y_mid = y_vals[idx_mid]

            norm_length = np.sqrt(w1**2 + w2**2)
            if norm_length > 0:
                scale = 0.4
                dx = w1 / norm_length * scale
                dy = w2 / norm_length * scale
                ax.arrow(x_mid, y_mid, dx, dy,
                        head_width=0.1, head_length=0.1,
                        fc='green', ec='green',
                        alpha=0.7, width=0.05)

    if len(weights) > 0:
        ax.legend(loc='best', fontsize=10)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--biases', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='')
    parser.add_argument('--affineTransformation', type=str, choices=['rotate', 'scale', 'shear'])
    parser.add_argument('--angle', type=float, default=30)
    parser.add_argument('--scale', type=float, default=1.5)
    parser.add_argument('--shear', type=float, default=0.3)

    args = parser.parse_args()

    weights_layer0, biases_layer0 = parse_weights_biases(args.weights, args.biases)

    df = pd.read_csv(args.dataset, header=None, names=['x', 'y', 'class', 'class_name'])
    df['x'] = pd.to_numeric(df['x'], errors='coerce')
    df['y'] = pd.to_numeric(df['y'], errors='coerce')
    df['class'] = pd.to_numeric(df['class'], errors='coerce')

    df = df.dropna()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    colors = ['blue' if cls == 0 else 'red' for cls in df['class']]

    ax1.scatter(df['x'], df['y'], c=colors, alpha=0.7, s=30, edgecolors='black', linewidth=0.5)
    ax1.set_title('Original Data')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')

    x_min, x_max = safe_limits(df['x'].values)
    y_min, y_max = safe_limits(df['y'].values)

    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)

    plot_decision_boundary(ax1, weights_layer0, biases_layer0, (x_min, x_max), (y_min, y_max))

    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')

    data_points = df[['x', 'y']].values
    if args.affineTransformation == 'rotate':
        transformed_data = apply_rotation(data_points, args.angle)
        transformed_weights = apply_rotation(np.array(weights_layer0), args.angle)
        ax2.set_title(f'Rotated Data (angle={args.angle}°)')

    elif args.affineTransformation == 'scale':
        transformed_data = apply_scale(data_points, args.scale, args.scale)
        transformed_weights = apply_scale(np.array(weights_layer0), args.scale, args.scale, False)
        ax2.set_title(f'Scaled Data (scale_x={args.scale}, scale_y={args.scale})')

    elif args.affineTransformation == 'shear':
        transformed_data = apply_shear(data_points, args.shear, args.shear)
        transformed_weights = apply_shear(np.array(weights_layer0), args.shear, args.shear, False)
        ax2.set_title(f'Sheared Data (shear_x={args.shear}, shear_y={args.shear})')

    ax2.scatter(transformed_data[:, 0], transformed_data[:, 1],
                c=colors, alpha=0.7, s=30, edgecolors='black', linewidth=0.5)

    x_min_t, x_max_t = safe_limits(transformed_data[:, 0])
    y_min_t, y_max_t = safe_limits(transformed_data[:, 1])

    ax2.set_xlim(x_min_t, x_max_t)
    ax2.set_ylim(y_min_t, y_max_t)

    plot_decision_boundary(ax2, transformed_weights, biases_layer0, (x_min_t, x_max_t), (y_min_t, y_max_t))

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()