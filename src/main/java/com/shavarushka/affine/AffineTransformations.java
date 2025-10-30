package com.shavarushka.affine;

public class AffineTransformations {

    public static double[][] rotate(double[][] weights, int axis1, int axis2, double angleDegrees) {
        return Rotation.rotate(weights, axis1, axis2, angleDegrees);
    }
}
