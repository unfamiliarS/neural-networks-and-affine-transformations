package com.shavarushka.affine;

public class Test {
    public static void main(String[] args) {
        AffineTransformation shareTransformation = new ShearAffineTransformation().shear(5);

        double[][] matrix = shareTransformation.createAffineMatrix(4);
        System.out.println();
        MatrixUtils.printMatrix(matrix);
    }
}
