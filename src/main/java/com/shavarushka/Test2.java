package com.shavarushka;

import java.io.File;
import java.io.IOException;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

import com.shavarushka.affine.AffineTransformations;
import com.shavarushka.network.api.ModelImagePredictorExtended;
import com.shavarushka.network.api.ModelLoader;
import com.shavarushka.network.api.WeightsManager;
import com.shavarushka.network.mnist.MNISTPredictor;

public class Test2 {
    public static void main(String[] args) throws IOException {
        MultiLayerNetwork network = ModelLoader.load("src/main/resources/mnist-model.zip");

        WeightsManager weightsManager = new WeightsManager(network);
        ModelImagePredictorExtended predictor = new MNISTPredictor(network);

        System.out.println("Before rotation");
        System.out.println(predictor.predictDetailedFromImage(new File("src/main/resources/mnist-nums/3_000003.png")));
        System.out.println(predictor.predictDetailedFromImage(new File("src/main/resources/mnist-nums/3_000003-180deg.png")));

        int layerIndex = 0;
        System.out.println();
        System.out.println(weightsManager.getLayerWeights(layerIndex)[0].length + " " + weightsManager.getLayerWeights(layerIndex).length);
        System.out.println();

        double rotationDegr = 180;
        double[][] firstHidenLayerWeights = weightsManager.getLayerWeights(layerIndex);
        double[][] rotatedWeights;
        rotatedWeights = AffineTransformations.rotate(firstHidenLayerWeights, 0, 1, rotationDegr);
        weightsManager.setLayerWeights(layerIndex, rotatedWeights);

        System.out.println("After rotation on " + rotationDegr);
        System.out.println(predictor.predictDetailedFromImage(new File("src/main/resources/mnist-nums/3_000003.png")));
        System.out.println(predictor.predictDetailedFromImage(new File("src/main/resources/mnist-nums/3_000003-180deg.png")));
    }
}
