package com.shavarushka.affine;

public class MatrixUtils {

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
