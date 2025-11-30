package com.shavarushka.affine;

public class RotationAffineTransformation extends AffineTransformation {

    private double angleDegrees;

    public RotationAffineTransformation angle(double angle) {
        angleDegrees = angle;
        return this;
    }

    @Override
    protected double[][] applyTransformationAroundAllAxis(double[][] coordinates) {
        int n = coordinates[0].length;

        double[][] affineMatrix = createAffineMatrix(n);

        double[][] transformed = MatrixUtils.multiplyWithTranspose(affineMatrix, coordinates);

        return transformed;
    }

    @Override
    protected double[][] createAffineMatrix(int dimensions) {
        double[][][] affineMatrices = new double[dimensions-1][dimensions][dimensions];

        for (int i = 0; i < dimensions-1; i++)
            affineMatrices[i] = createSimpleAffineMatrix(dimensions, i, i+1);

        double[][] result = affineMatrices[0];
        for (int i = 1; i < dimensions-1; i++)
            result = MatrixUtils.multiplyMatrices(result, affineMatrices[i]);

        return result;
    }

    private double[][] createSimpleAffineMatrix(int dimensions, int axis1, int axis2) {
        double[][] matrix = createIdentityMatrix(dimensions);

        double cos = Math.cos(angleDegrees);
        double sin = Math.sin(angleDegrees);

        matrix[axis1][axis1] = cos;
        matrix[axis1][axis2] = -sin;
        matrix[axis2][axis1] = sin;
        matrix[axis2][axis2] = cos;

        return matrix;
    }
}
