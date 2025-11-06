package com.shavarushka;

import com.shavarushka.affine.AffineTransformations;
import com.shavarushka.affine.MatrixUtils;

public class Test3 {
    public static void main(String[] args) {
        double[][] testImage = {
            {1, 0},
            {0, 1},
            {0, 0}
        };

        double[][] rotatedTest = AffineTransformations.rotate(testImage, 0, 1, 180);
        MatrixUtils.printMatrix(rotatedTest);
    }
}
