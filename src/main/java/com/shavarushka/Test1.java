package com.shavarushka;

import java.io.IOException;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

import com.shavarushka.affine.AffineTransformations;
import com.shavarushka.network.api.ModelLoader;
import com.shavarushka.network.api.WeightsManager;

public class Test1 {
    public static void main(String[] args) throws IOException {
        MultiLayerNetwork network = ModelLoader.load("src/main/resources/triangle-classifier.zip");

        WeightsManager weightsManager = new WeightsManager(network);
        weightsManager.printWeights();

        int layerIndex = 0;
        double rotationDegr = 180;
        double[][][] allWeights = weightsManager.getAllWeights();
        double[][] rotatedWeights;
        rotatedWeights = AffineTransformations.rotate(allWeights[layerIndex], 0, 1, rotationDegr);
        weightsManager.setLayerWeights(layerIndex, rotatedWeights);

        System.out.println();
        weightsManager.printWeights();
    }
}
