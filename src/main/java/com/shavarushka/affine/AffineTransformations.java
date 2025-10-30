package com.shavarushka.affine;

public class AffineTransformations {

    public static double[][] rotate(double[][] weights, int axis1, int axis2, double angleDegrees) {
        return Rotation.rotate(weights, axis1, axis2, angleDegrees);
    }

    public static double[][] scale(double[][] weights, double scaleX, double scaleY) {
        return Scale.scale(weights, scaleX, scaleY);
    }

    public static double[][] scale(double[][] weights, double scale) {
        return Scale.scale(weights, scale);
    }
}
