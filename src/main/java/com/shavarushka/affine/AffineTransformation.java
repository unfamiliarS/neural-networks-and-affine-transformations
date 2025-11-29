package com.shavarushka.affine;

public class AffineTransformation {

    private AffineMatrixProvider matrixProvider;

    public AffineTransformation(AffineMatrixProvider matrixProvider) {
        this.matrixProvider = matrixProvider;
    }

    public void setMatrixProvider(AffineMatrixProvider provider) {
        matrixProvider = provider;
    }

    public double[][] transform(double[][] matrix, int axis1, int axis2) {
        if (matrix == null || matrix.length == 0)
            return new double[0][0];

        double[][] transformedMatrix = applyTransformation(matrix, axis1, axis2);

        return transformedMatrix;
    }

    private double[][] applyTransformation(double[][] coordinates, int axis1, int axis2) {
        int n = coordinates[0].length;

        if (axis1 < 0 || axis1 >= n || axis2 < 0 || axis2 >= n || axis1 == axis2)
            throw new IllegalArgumentException("Invalid rotation axes");

        double[][] affineMatrix = matrixProvider.createAffineMatrix(n, axis1, axis2);

        double[][] transformed = MatrixUtils.multiplyWithTranspose(affineMatrix, coordinates);

        return transformed;
    }

    public double[][] transformComplex(double[][] matrix) {
        if (matrix == null || matrix.length == 0)
            return new double[0][0];

        double[][] transformedMatrix = applyTransformationAroundAllAxis(matrix);

        return transformedMatrix;
    }

    private double[][] applyTransformationAroundAllAxis(double[][] coordinates) {
        int n = coordinates[0].length;

        double[][] affineMatrix = createComplexAffineMatrix(n);

        double[][] transformed = MatrixUtils.multiplyWithTranspose(affineMatrix, coordinates);

        return transformed;
    }

    private double[][] createComplexAffineMatrix(int dimensions) {
        double[][][] affineMatrices = new double[dimensions-1][dimensions][dimensions];

        for (int i = 0; i < dimensions-1; i++) {
            affineMatrices[i] = matrixProvider.createAffineMatrix(dimensions, i, i+1);
        }

        double[][] result = affineMatrices[0];
        for (int i = 1; i < dimensions - 1; i++)
            result = MatrixUtils.multiplyMatrices(result, affineMatrices[i]);

        return result;
    }
}
