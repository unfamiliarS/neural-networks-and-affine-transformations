package com.shavarushka.affine;

public class ShearAffineTransformation extends AffineTransformation {

    private double shear;
    private boolean isData;

    public ShearAffineTransformation shear(double shear) {
        this.shear = shear;
        return this;
    }

    public ShearAffineTransformation setMatrixType(boolean isData) {
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
    protected double[][] createAffineMatrix(int dimensions) {
        double[][] result = createIdentityMatrix(dimensions);

        for (int i = 0; i < dimensions-1; i++) {
            double[][] rotationMatrix = createSimpleAffineMatrix(dimensions, i, i+1);
            result = MatrixUtils.multiplyMatrices(result, rotationMatrix);
        }

        // for (int i = 0; i < dimensions; i++) {
        //     for (int j = 0; j < dimensions; j++) {
        //         if (i != j) {
        //             double[][] rotationMatrix = createSimpleAffineMatrix(dimensions, i, j);
        //             result = MatrixUtils.multiplyMatrices(result, rotationMatrix);
        //         }
        //     }
        // }

        // for (int i = 0; i < dimensions; i++) {
        //     for (int j = i+1; j < dimensions; j += 5) {
        //         double[][] rotationMatrix = createSimpleAffineMatrix(dimensions, i, j);
        //         result = MatrixUtils.multiplyMatrices(result, rotationMatrix);
        //     }
        // }

        // double[][] rotationMatrix;
        // for (int i = 0; i < dimensions; i++) {
        //     if (i == 0)
        //         rotationMatrix = createSimpleAffineMatrix(dimensions, i, i+1);
        //     else
        //         rotationMatrix = createSimpleAffineMatrix(dimensions, i, 0);

        //     result = MatrixUtils.multiplyMatrices(result, rotationMatrix);
        // }

        return result;
    }

    private double[][] createSimpleAffineMatrix(int dimensions, int axis1, int axis2) {
        double[][] matrix = createIdentityMatrix(dimensions);

        matrix[axis1][axis2] = shear;
        matrix[axis2][axis1] = shear;

        return matrix;
    }
}
