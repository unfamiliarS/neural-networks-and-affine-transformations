package com.shavarushka.affine;

public class Test {
    public static void main(String[] args) {
        double[][] weights = {
            {-3.262287139892578, -2.8782620429992676},
            {1.9234658479690552, -1.6480084657669067}
        };
        
        System.out.println("Original matrix:");
        MatrixUtils.printMatrix(weights);
        
        double[][] scale5x5 = Scale.scale(weights, 5.0);
        System.out.println("\nScale 5x5:");
        MatrixUtils.printMatrix(scale5x5);
    }
}