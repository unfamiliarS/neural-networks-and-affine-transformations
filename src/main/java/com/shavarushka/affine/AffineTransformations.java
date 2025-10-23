package com.shavarushka.affine;

public class AffineTransformations {

    public static double[][] rotate(double[][] weights, double angleDegrees) {
        return Rotation.rotate(weights, angleDegrees);
    }

    public static double[][] scale(double[][] weights, double scaleX, double scaleY) {
        return Scale.scale(weights, scaleX, scaleY);
    }

    public static double[][] scale(double[][] weights, double scale) {
        return Scale.scale(weights, scale);
    }    
}
