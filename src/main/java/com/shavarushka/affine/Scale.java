package com.shavarushka.affine;

public class Scale {

    /**
     * Растягивает/сжимает матрицу весов по осям
     * @param weights матрица весов [rows][cols]
     * @param scaleX коэффициент масштабирования по X
     * @param scaleY коэффициент масштабирования по Y
     * @return преобразованная матрица весов
     */
    public static double[][] scale(double[][] weights, double scaleX, double scaleY) {
        if (weights == null || weights.length == 0) {
            throw new IllegalArgumentException("Weights matrix cannot be null or empty");
        }

        if (Math.abs(scaleX) < 1e-10 || Math.abs(scaleY) < 1e-10) {
            throw new IllegalArgumentException("Scale factors cannot be zero");
        }

        // Матрица масштабирования
        double[][] scaleMatrix = {
            {scaleX, 0},
            {0, scaleY}
        };

        // Инвертируем матрицу масштабирования
        double[][] invScaleMatrix = invert2x2Matrix(scaleMatrix);

        // Транспонируем обратную матрицу масштабирования
        double[][] invScaleMatrixTransposed = transpose(invScaleMatrix);

        // Применяем преобразование: W' = S^(-T) * W
        return multiplyMatrices(invScaleMatrixTransposed, weights);
    }

    /**
     * Растягивает матрицу весов равномерно по обеим осям
     */
    public static double[][] scale(double[][] weights, double scale) {
        return scale(weights, scale, scale);
    }

    private static double[][] invert2x2Matrix(double[][] matrix) {
        if (matrix.length != 2 || matrix[0].length != 2) {
            throw new IllegalArgumentException("Matrix must be 2x2");
        }

        double a = matrix[0][0];
        double b = matrix[0][1];
        double c = matrix[1][0];
        double d = matrix[1][1];

        double determinant = a * d - b * c;

        if (Math.abs(determinant) < 1e-10) {
            throw new IllegalArgumentException("Matrix is not invertible");
        }

        double invDet = 1.0 / determinant;

        return new double[][] {
            {d * invDet, -b * invDet},
            {-c * invDet, a * invDet}
        };
    }

    private static double[][] transpose(double[][] matrix) {
        int rows = matrix.length;
        int cols = matrix[0].length;

        double[][] result = new double[cols][rows];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[j][i] = matrix[i][j];
            }
        }

        return result;
    }

    private static double[][] multiplyMatrices(double[][] a, double[][] b) {
        int aRows = a.length;
        int aCols = a[0].length;
        int bRows = b.length;
        int bCols = b[0].length;

        if (aCols != bRows) {
            throw new IllegalArgumentException(
                "Matrix multiplication not possible: " + aCols + " != " + bRows);
        }

        double[][] result = new double[aRows][bCols];

        for (int i = 0; i < aRows; i++) {
            for (int j = 0; j < bCols; j++) {
                double sum = 0.0;
                for (int k = 0; k < aCols; k++) {
                    sum += a[i][k] * b[k][j];
                }
                result[i][j] = sum;
            }
        }

        return result;
    }
}
