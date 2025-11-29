package com.shavarushka.affine;

public class RotationMatrixProvider implements AffineMatrixProvider {

    private double angleDegrees;

    public RotationMatrixProvider setAngle(double angle) {
        angleDegrees = angle;
        return this;
    }

    @Override
    public double[][] createAffineMatrix(int dimensions, int axis1, int axis2) {
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
