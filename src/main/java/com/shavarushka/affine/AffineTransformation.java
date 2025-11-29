package com.shavarushka.affine;

public abstract class AffineTransformation {

    public double[][] transform(double[][] matrix) {
        if (matrix == null || matrix.length == 0)
            return new double[0][0];

        double[][] transformedMatrix = applyTransformationAroundAllAxis(matrix);

        return transformedMatrix;
    }

    protected abstract double[][] applyTransformationAroundAllAxis(double[][] coordinates);
    protected abstract double[][] createAffineMatrix(int dimensions);

    protected double[][] createIdentityMatrix(int dimensions) {
        double[][] matrix = new double[dimensions][dimensions];

        for (int i = 0; i < dimensions; i++)
            for (int j = 0; j < dimensions; j++)
                matrix[i][j] = (i == j) ? 1.0 : 0.0;

        return matrix;
    }
}
