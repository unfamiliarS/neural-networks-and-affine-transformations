package com.shavarushka.affine;

import org.ejml.simple.SimpleMatrix;

public class MatrixUtils {

    public static double[][] multiplyMatrices(double[][] a, double[][] b) {
        SimpleMatrix matrixA = new SimpleMatrix(a);
        SimpleMatrix matrixB = new SimpleMatrix(b);
        return matrixA.mult(matrixB).toArray2();
    }

    public static double[][] inverse(double[][] matrix) {
        SimpleMatrix simpleMatrix = new SimpleMatrix(matrix);
        return simpleMatrix.invert().toArray2();
    }

    public static double[][] multiplyWithTranspose(double[][] rotationMatrix, double[][] coordinates) {
        SimpleMatrix rot = new SimpleMatrix(rotationMatrix);
        SimpleMatrix coords = new SimpleMatrix(coordinates);

        SimpleMatrix result = rot.mult(coords.transpose());
        
        return result.transpose().toArray2();
    }

    public static void printMatrix(double[][] matrix) {
        if (matrix == null) {
            System.out.println("Matrix is null");
            return;
        }

        for (double[] row : matrix) {
            for (double value : row) {
                System.out.print(value + " ");
            }
            System.out.println();
        }
    }
}
