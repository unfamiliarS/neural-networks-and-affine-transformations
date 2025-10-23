package com.shavarushka.network;

import com.shavarushka.affine.AffineTransformations;

public class Main {
    public static void main(String[] args) {
        MNISTClassifier classifier = MNISTClassifier.load("src/main/resources/mnist-model.zip");

        double[][][] allWeights = classifier.getWeights();
        
        double[][] rotatedWeights;
        for (int i = 0; i < allWeights.length; i++) {
            rotatedWeights = AffineTransformations.rotate(allWeights[i], 90);
            classifier.setWeights(i, rotatedWeights);
        }

        classifier.predict(null);
    }
}
