package com.shavarushka.affine;

public interface AffineMatrixProvider {
    double[][] createAffineMatrix(int dimensions, int axis1, int axis2);

    default double[][] createIdentityMatrix(int dimensions) {
        double[][] matrix = new double[dimensions][dimensions];

        for (int i = 0; i < dimensions; i++)
            for (int j = 0; j < dimensions; j++)
                matrix[i][j] = (i == j) ? 1.0 : 0.0;

        return matrix;
    }
}
