package com.shavarushka.affine;

public class Rotation {
    
    /**
     * Поворачивает матрицу весов на заданный угол
     * @param weights матрица весов [rows][cols]
     * @param angleDegrees угол поворота в градусах
     * @return повернутая матрица весов
     */
    public static double[][] rotate(double[][] weights, double angleDegrees) {
        if (weights == null || weights.length == 0) {
            throw new IllegalArgumentException("Weights matrix cannot be null or empty");
        }
        
        // Преобразуем угол в радианы
        double angleRad = Math.toRadians(angleDegrees);
        double cos = Math.cos(angleRad);
        double sin = Math.sin(angleRad);
        
        // Матрица поворота
        double[][] rotationMatrix = {
            {cos, sin},
            {-sin, cos}
        };
        
        // Для поворота матрицы весов применяем обратное преобразование
        double[][] invRotationMatrix = invert2x2Matrix(rotationMatrix);
        
        // Транспонируем обратную матрицу поворота
        double[][] invRotationMatrixTransposed = transpose(invRotationMatrix);
        
        // Применяем преобразование: W' = R^(-T) * W
        return multiplyMatrices(invRotationMatrixTransposed, weights);
    }
    
    /**
     * Инвертирует матрицу 2x2
     */
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
    
    /**
     * Транспонирует матрицу
     */
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
    
    /**
     * Умножает две матрицы
     */
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
