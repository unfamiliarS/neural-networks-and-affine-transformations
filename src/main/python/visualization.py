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

def apply_scale(data, scale_x, scale_y):
    scale_matrix = np.array([
        [scale_x, 0],
        [0, scale_y]
    ])
    result = scale_matrix @ data.T
    return result.T

def apply_shear(data, shear_x, shear_y):
    shear_matrix = np.array([
        [1, shear_x],
        [shear_y, 1]
    ])
    result = shear_matrix @ data.T
    return result.T

def transform_weights(weights, transformation, **kwargs):
    transformed_weights = []
    for neuron_weights in weights:
        weights_array = np.array(neuron_weights)
        if transformation == 'rotate':
            angle_rad = np.radians(kwargs['angle'])
            rotation_matrix = np.array([
                [np.cos(angle_rad), -np.sin(angle_rad)],
                [np.sin(angle_rad), np.cos(angle_rad)]
            ])
            transformed = rotation_matrix @ weights_array
        elif transformation == 'scale':
            scale_matrix = np.array([
                [1/kwargs['scale_x'], 0],
                [0, 1/kwargs['scale_y']]
            ])
            transformed = weights_array @ scale_matrix
        elif transformation == 'shear':
            shear_matrix = np.array([
                [1, -kwargs['shear_x']],
                [-kwargs['shear_y'], 1]
            ])
            transformed = weights_array @ shear_matrix
        transformed_weights.append(transformed.tolist())
    return transformed_weights

def safe_limits(data):
    clean_data = data[np.isfinite(data)]
    if len(clean_data) == 0:
        return -1, 1
    data_min = np.min(clean_data)
    data_max = np.max(clean_data)
    padding = (data_max - data_min) * 0.1
    return data_min - padding, data_max + padding

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--biases', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='/home/semyon/projects/neural-networks-and-affine-transformations/src/main/python/multipletriangle/dataset.csv')
    parser.add_argument('--output', type=str, default='multiple-triangles-class.png')
    parser.add_argument('--affineTransformation', type=str, choices=['rotate', 'scale', 'shear'])
    parser.add_argument('--angle', type=float, default=45)
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

    # Original data
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

    # Transformed data
    if args.affineTransformation:
        data_points = df[['x', 'y']].values
        
        if args.affineTransformation == 'rotate':
            transformed_data = apply_rotation(data_points, args.angle)
            transformed_weights = transform_weights(weights_layer0, 'rotate', angle=args.angle)
            ax2.set_title(f'Rotated Data (angle={args.angle}°)')
            
        elif args.affineTransformation == 'scale':
            transformed_data = apply_scale(data_points, args.scale, args.scale)
            transformed_weights = transform_weights(weights_layer0, 'scale', scale_x=args.scale, scale_y=args.scale)
            ax2.set_title(f'Scaled Data (scale_x={args.scale}, scale_y={args.scale})')
            
        elif args.affineTransformation == 'shear':
            transformed_data = apply_shear(data_points, args.shear, args.shear)
            transformed_weights = transform_weights(weights_layer0, 'shear', shear_x=args.shear, shear_y=args.shear)
            ax2.set_title(f'Sheared Data (shear_x={args.shear}, shear_y={args.shear})')
        
        ax2.scatter(transformed_data[:, 0], transformed_data[:, 1], c=colors, alpha=0.7, s=30, edgecolors='black', linewidth=0.5)
        
        x_min_t, x_max_t = safe_limits(transformed_data[:, 0])
        y_min_t, y_max_t = safe_limits(transformed_data[:, 1])
        
        ax2.set_xlim(x_min_t, x_max_t)
        ax2.set_ylim(y_min_t, y_max_t)
        
        x_range_t = np.linspace(x_min_t, x_max_t, 200)
        
        for neuron_idx in range(len(transformed_weights)):
            w1 = transformed_weights[neuron_idx][0]
            w2 = transformed_weights[neuron_idx][1]
            b = biases_layer0[neuron_idx]

            if abs(w2) > 1e-6:
                y_line = (-w1 * x_range_t - b) / w2
                valid_indices = (y_line >= y_min_t) & (y_line <= y_max_t)
                if np.any(valid_indices):
                    ax2.plot(x_range_t[valid_indices], y_line[valid_indices], linewidth=2, alpha=0.8)
    else:
        ax2.scatter(df['x'], df['y'], c=colors, alpha=0.7, s=30, edgecolors='black', linewidth=0.5)
        ax2.set_title('No Transformation')
        transformed_weights = weights_layer0
        ax2.set_xlim(x_min, x_max)
        ax2.set_ylim(y_min, y_max)

    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')

    x_range = np.linspace(x_min, x_max, 200)

    for neuron_idx in range(len(weights_layer0)):
        w1 = weights_layer0[neuron_idx][0]
        w2 = weights_layer0[neuron_idx][1]
        b = biases_layer0[neuron_idx]

        if abs(w2) > 1e-6:
            y_line = (-w1 * x_range - b) / w2
            valid_indices = (y_line >= y_min) & (y_line <= y_max)
            if np.any(valid_indices):
                ax1.plot(x_range[valid_indices], y_line[valid_indices], linewidth=2, alpha=0.8)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
