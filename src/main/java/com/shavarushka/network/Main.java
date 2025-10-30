package com.shavarushka.network;

import java.io.File;
import java.io.IOException;

import com.shavarushka.network.api.ModelImagePredictorExtended;
import com.shavarushka.network.api.ModelNetwork;
import com.shavarushka.network.api.PredictionResult;
import com.shavarushka.network.mnist.MNISTPredictor;

public class Main {
    public static void main(String[] args) throws IOException {
        ModelNetwork network = ModelNetwork.load("src/main/resources/mnist-model.zip");
        
        ModelImagePredictorExtended predictor = new MNISTPredictor(network.getModel());
        PredictionResult result = predictor.predictDetailedFromImage(new File("src/main/resources/mnist-nums/8_005839.png"));
        result.printDetails();
    }
}
