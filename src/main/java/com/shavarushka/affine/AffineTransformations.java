package com.shavarushka.affine;

public class AffineTransformations {

    public static double[][] rotate(double[][] matrix, int axis1, int axis2, double angleDegrees) {
        return Rotation.rotate(matrix, axis1, axis2, angleDegrees);
    }
   
    public static double[][] rotateComplex(double[][] matrix, double angleDegrees) {
        return Rotation.rotateAroundAllAxis(matrix, angleDegrees);
    }
}
