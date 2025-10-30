package com.shavarushka;

import java.io.File;
import java.io.IOException;

import com.shavarushka.affine.AffineTransformations;
import com.shavarushka.network.api.ModelImagePredictorExtended;
import com.shavarushka.network.api.ModelNetwork;
import com.shavarushka.network.api.WeightsManager;
import com.shavarushka.network.mnist.MNISTPredictor;

public class Main {
    public static void main(String[] args) throws IOException {
        ModelNetwork network = ModelNetwork.load("src/main/resources/mnist-model.zip");
        WeightsManager weightsManager = new WeightsManager(network.getModel());
        ModelImagePredictorExtended predictor = new MNISTPredictor(network.getModel());

        System.out.println("Before rotation");
        System.out.println(predictor.predictDetailedFromImage(new File("src/main/resources/mnist-nums/3_000003.png")));
        System.out.println(predictor.predictDetailedFromImage(new File("src/main/resources/mnist-nums/3_000003-180deg.png")));

        double rotationDegr = 360;
        double[][][] allWeights = weightsManager.getAllWeights();
        double[][] rotatedWeights;
        for (int i = 0; i < allWeights.length; i++) {
            rotatedWeights = AffineTransformations.rotate(allWeights[i], 8, 12, rotationDegr);
            weightsManager.setLayerWeights(i, rotatedWeights);
        }

        System.out.println("After rotation on " + rotationDegr);
        System.out.println(predictor.predictDetailedFromImage(new File("src/main/resources/mnist-nums/3_000003.png")));
        System.out.println(predictor.predictDetailedFromImage(new File("src/main/resources/mnist-nums/3_000003-180deg.png")));
    }
}
