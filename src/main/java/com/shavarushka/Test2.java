package com.shavarushka;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

import com.shavarushka.affine.AffineTransformations;
import com.shavarushka.network.api.ModelLoader;
import com.shavarushka.network.api.WeightsManager;

public class Test2 {
    public static void main(String[] args) {
        MultiLayerNetwork network = ModelLoader.load("src/main/resources/triangle-classifier.zip");
        WeightsManager weightsManager = new WeightsManager(network);
        weightsManager.printWeights();
        
        double[][] weights = weightsManager.getLayerWeights(0);
        double[][] rotatedWeights = AffineTransformations.rotate(weights, 0, 1, 180);
     
        // network = ModelLoader.load("src/main/resources/mnist-model.zip");
        // weightsManager = new WeightsManager(network);
        
        // weights = weightsManager.getLayerWeights(3);
        // rotatedWeights = AffineTransformations.rotate(weights, 0, 1, 180);        

    }
}
