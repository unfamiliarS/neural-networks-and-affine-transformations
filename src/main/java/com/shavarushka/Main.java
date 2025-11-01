package com.shavarushka;

import java.io.File;
import java.io.IOException;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

import com.shavarushka.affine.AffineTransformations;
import com.shavarushka.network.api.DataIterators;
import com.shavarushka.network.api.IteratorDataEvaluator;
import com.shavarushka.network.api.IteratorDataTrainer;
import com.shavarushka.network.api.ModelEvaluator;
import com.shavarushka.network.api.ModelImagePredictorExtended;
import com.shavarushka.network.api.ModelLoader;
import com.shavarushka.network.api.ModelTrainer;
import com.shavarushka.network.api.WeightsManager;
import com.shavarushka.network.mnist.MNISTDataIterators;
import com.shavarushka.network.mnist.MNISTNetwork;
import com.shavarushka.network.mnist.MNISTPredictor;

public class Main {
    public static void main(String[] args) throws IOException {
        MultiLayerNetwork network = ModelLoader.load("src/main/resources/mnist-model.zip");

        WeightsManager weightsManager = new WeightsManager(network);
        ModelImagePredictorExtended predictor = new MNISTPredictor(network);

        System.out.println("Before rotation");
        System.out.println(predictor.predictDetailedFromImage(new File("src/main/resources/mnist-nums/3_000003.png")));
        System.out.println(predictor.predictDetailedFromImage(new File("src/main/resources/mnist-nums/3_000003-180deg.png")));

        double rotationDegr = 180;
        double[][][] allWeights = weightsManager.getAllWeights();
        double[][] rotatedWeights;
        for (int i = 0; i < allWeights.length; i++) {
            rotatedWeights = AffineTransformations.rotate(allWeights[i], 0, 1, rotationDegr);
            weightsManager.setLayerWeights(i, rotatedWeights);
        }

        System.out.println("After rotation on " + rotationDegr);
        System.out.println(predictor.predictDetailedFromImage(new File("src/main/resources/mnist-nums/3_000003.png")));
        System.out.println(predictor.predictDetailedFromImage(new File("src/main/resources/mnist-nums/3_000003-180deg.png")));
    }
}
