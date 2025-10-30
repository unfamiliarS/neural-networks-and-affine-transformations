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

    public static double[][] createSequentialMatrix(int rows, int cols) {
        double[][] matrix = new double[rows][cols];
        double value = 1.0;

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                matrix[i][j] = value++;
            }
        }

        return matrix;
    }

    public static double[][] createConstantMatrix(int rows, int cols, double value) {
        double[][] matrix = new double[rows][cols];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                matrix[i][j] = value;
            }
        }

        return matrix;
    }
}
