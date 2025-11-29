package com.shavarushka.affine;

public class ScaleMatrixProvider implements AffineMatrixProvider {

    private double scaleFactor;
    private boolean uniformScaling = true;
    private double[] scaleFactors;

    public ScaleMatrixProvider setScaleFactor(double scaleFactor) {
        this.scaleFactor = scaleFactor;
        return this;
    }

    public ScaleMatrixProvider setScaleFactors(double[] scaleFactors) {
        this.scaleFactors = scaleFactors.clone();
        this.uniformScaling = false;
        return this;
    }

    @Override
    public double[][] createAffineMatrix(int dimensions, int axis1, int axis2) {
        double[][] matrix = createIdentityMatrix(dimensions);

        if (uniformScaling) {
            for (int i = 0; i < dimensions; i++)
                matrix[i][i] = scaleFactor;
        } else {
            for (int i = 0; i < dimensions; i++)
                matrix[i][i] = i < scaleFactors.length ? scaleFactors[i] : 1.0;
        }

        return matrix;
    }
}
