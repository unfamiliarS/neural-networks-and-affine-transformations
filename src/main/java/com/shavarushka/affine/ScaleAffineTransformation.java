package com.shavarushka.affine;

public class ScaleAffineTransformation extends AffineTransformation {

    private double scaleFactor;
    private boolean isData;

    public ScaleAffineTransformation scaleFactor(double scaleFactor) {
        this.scaleFactor = scaleFactor;
        return this;
    }

    public ScaleAffineTransformation setMatrixType(boolean isData) {
        this.isData = isData;
        return this;
    }

    @Override
    protected double[][] applyTransformationAroundAllAxis(double[][] coordinates) {
        int n = coordinates[0].length;

        double[][] affineMatrix = createAffineMatrix(n);

        double[][] transformed = !isData
            ? MatrixUtils.multiplyWithTranspose(MatrixUtils.inverse(affineMatrix), coordinates)
            : MatrixUtils.multiplyWithTranspose(affineMatrix, coordinates);

        return transformed;
    }

    @Override
    public double[][] createAffineMatrix(int dimensions) {
        double[][] matrix = createIdentityMatrix(dimensions);

        for (int i = 0; i < dimensions; i++)
            matrix[i][i] = scaleFactor;

        return matrix;
    }
}
