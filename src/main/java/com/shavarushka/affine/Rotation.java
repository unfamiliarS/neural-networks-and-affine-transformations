package com.shavarushka.affine;

public class Rotation {

    public static double[][] rotate(double[][] weightMatrix, int axis1, int axis2, double angleDegrees) {
        if (weightMatrix == null || weightMatrix.length == 0) {
            return new double[0][0];
        }

        double[][] hyperplanes = extractHyperplanes(weightMatrix);

        double angleRadians = Math.toRadians(angleDegrees);
        double[][] rotatedHyperplanes = applyRotation(hyperplanes, axis1, axis2, angleRadians);

        return hyperplanesToMatrix(rotatedHyperplanes);
    }

    private static double[][] extractHyperplanes(double[][] weightMatrix) {
        int numRows = weightMatrix.length;
        int numCols = weightMatrix[0].length;

        double[][] hyperplanes = new double[numCols][numRows];

        for (int i = 0; i < numRows; i++) {
            for (int j = 0; j < numCols; j++) {
                hyperplanes[j][i] = weightMatrix[i][j];
            }
        }

        return hyperplanes;
    }

    private static double[][] hyperplanesToMatrix(double[][] hyperplanes) {
        int numPlanes = hyperplanes.length;
        int numCoords = hyperplanes[0].length;

        double[][] weightMatrix = new double[numCoords][numPlanes];

        for (int i = 0; i < numCoords; i++) {
            for (int j = 0; j < numPlanes; j++) {
                weightMatrix[i][j] = hyperplanes[j][i];
            }
        }

        return weightMatrix;
    }

    private static double[][] applyRotation(double[][] coordinates, int axis1, int axis2, double angle) {
        int n = coordinates[0].length;
        int points = coordinates.length;

        if (axis1 < 0 || axis1 >= n || axis2 < 0 || axis2 >= n || axis1 == axis2) {
            throw new IllegalArgumentException("Invalid rotation axes");
        }

        double[][] rotationMatrix = createRotationMatrix(n, axis1, axis2, angle);

        double[][] rotated = new double[points][n];

        for (int p = 0; p < points; p++) {
            rotated[p] = multiplyMatrixVector(rotationMatrix, coordinates[p]);
        }

        return rotated;
    }

    private static double[][] createRotationMatrix(int dimensions, int axis1, int axis2, double angle) {
        double[][] matrix = new double[dimensions][dimensions];

        for (int i = 0; i < dimensions; i++) {
            for (int j = 0; j < dimensions; j++) {
                matrix[i][j] = (i == j) ? 1.0 : 0.0;
            }
        }

        double cos = Math.cos(angle);
        double sin = Math.sin(angle);

        matrix[axis1][axis1] = cos;
        matrix[axis1][axis2] = -sin;
        matrix[axis2][axis1] = sin;
        matrix[axis2][axis2] = cos;

        return matrix;
    }

    private static double[] multiplyMatrixVector(double[][] matrix, double[] vector) {
        int n = vector.length;
        double[] result = new double[n];

        for (int i = 0; i < n; i++) {
            double sum = 0.0;
            for (int j = 0; j < n; j++) {
                sum += matrix[i][j] * vector[j];
            }
            result[i] = sum;
        }

        return result;
    }
}