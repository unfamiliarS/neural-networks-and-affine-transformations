package com.shavarushka.affine;

import org.ejml.simple.SimpleMatrix;

class Rotation {

    static double[][] rotate(double[][] matrix, int axis1, int axis2, double angleDegrees) {
        if (matrix == null || matrix.length == 0)
            return new double[0][0];

        double angleRadians = Math.toRadians(angleDegrees);
        double[][] rotatedMatrix = applyRotation(matrix, axis1, axis2, angleRadians);

        return rotatedMatrix;
    }

    static double[][] rotateAroundAllAxis(double[][] matrix, double angleDegrees) {
        if (matrix == null || matrix.length == 0)
            return new double[0][0];

        double angleRadians = Math.toRadians(angleDegrees);
        double[][] rotatedMatrix = applyRotationAroundAllAxis(matrix, angleRadians);

        return rotatedMatrix;
    }

    private static double[][] applyRotationAroundAllAxis(double[][] coordinates, double angle) {
        int n = coordinates[0].length;
        int points = coordinates.length;

        double[][] rotationMatrix = createComplexRotationMatrix(n, angle);

        double[][] rotated = new double[points][n];

        for (int p = 0; p < points; p++)
            rotated[p] = multiplyMatrixVector(rotationMatrix, coordinates[p]);

        return rotated;
    }

    private static double[][] createComplexRotationMatrix(int dimensions, double angle) {
        double[][][] rotationMatrices = new double[dimensions-1][dimensions][dimensions];
        
        for (int i = 0; i < dimensions-1; i++) {
            // System.out.println("Create rotation matrix around axis: " + i + " " + i+1);
            rotationMatrices[i] = createRotationMatrix(dimensions, i, i+1, angle);
        }
        
        double[][] result = rotationMatrices[0];
        for (int i = 1; i < dimensions - 1; i++) {
            // System.out.println("Multiply rotation matrixes: " + i);
            result = multiplyMatrices(result, rotationMatrices[i]);
        }

        return result;
    }

    private static double[][] applyRotation(double[][] coordinates, int axis1, int axis2, double angle) {
        int n = coordinates[0].length;
        int points = coordinates.length;

        if (axis1 < 0 || axis1 >= n || axis2 < 0 || axis2 >= n || axis1 == axis2)
            throw new IllegalArgumentException("Invalid rotation axes");

        double[][] rotationMatrix = createRotationMatrix(n, axis1, axis2, angle);

        double[][] rotated = new double[points][n];

        for (int p = 0; p < points; p++)
            rotated[p] = multiplyMatrixVector(rotationMatrix, coordinates[p]);

        return rotated;
    }

    private static double[][] createRotationMatrix(int dimensions, int axis1, int axis2, double angle) {
        double[][] matrix = new double[dimensions][dimensions];

        for (int i = 0; i < dimensions; i++)
            for (int j = 0; j < dimensions; j++)
                matrix[i][j] = (i == j) ? 1.0 : 0.0;

        double cos = Math.cos(angle);
        double sin = Math.sin(angle);

        matrix[axis1][axis1] = cos;
        matrix[axis1][axis2] = -sin;
        matrix[axis2][axis1] = sin;
        matrix[axis2][axis2] = cos;

        return matrix;
    }

    private static double[][] multiplyMatrices(double[][] a, double[][] b) {
        SimpleMatrix matrixA = new SimpleMatrix(a);
        SimpleMatrix matrixB = new SimpleMatrix(b);
        return matrixA.mult(matrixB).toArray2();
    }

    // private static double[][] multiplyMatrices(double[][] a, double[][] b) {
    //     int n = a.length;
    //     double[][] result = new double[n][n];
        
    //     for (int i = 0; i < n; i++) {
    //         for (int j = 0; j < n; j++) {
    //             double sum = 0.0;
    //             for (int k = 0; k < n; k++) {
    //                 sum += a[i][k] * b[k][j];
    //             }
    //             result[i][j] = sum;
    //         }
    //     }
        
    //     return result;
    // }

    private static double[] multiplyMatrixVector(double[][] matrix, double[] vector) {
        int n = vector.length;
        double[] result = new double[n];

        for (int i = 0; i < n; i++) {
            double sum = 0.0;
            for (int j = 0; j < n; j++)
                sum += matrix[i][j] * vector[j];
   
            result[i] = sum;
        }

        return result;
    }
}