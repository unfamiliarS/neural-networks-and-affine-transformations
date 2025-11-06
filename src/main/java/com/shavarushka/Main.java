package com.shavarushka;

import java.io.File;
import java.io.IOException;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.cpu.nativecpu.bindings.Nd4jCpu.in_top_k;

import com.shavarushka.affine.AffineTransformations;
import com.shavarushka.affine.MatrixUtils;
import com.shavarushka.network.api.ModelImagePredictorExtended;
import com.shavarushka.network.api.ModelLoader;
import com.shavarushka.network.api.WeightsManager;
import com.shavarushka.network.mnist.MNISTPredictor;

public class Main {
    public static void main(String[] args) throws IOException {
        MultiLayerNetwork network = ModelLoader.load("src/main/resources/mnist-model.zip");

        WeightsManager weightsManager = new WeightsManager(network);
        ModelImagePredictorExtended predictor = new MNISTPredictor(network);

        double rotationDegr = 180;

        double[][] imageData = predictor.load(new File("src/main/resources/mnist-nums/5_000021.png"));
        double[][] rotatedImageData = AffineTransformations.rotate(imageData, 4, 5, rotationDegr);

        MatrixUtils.printMatrix(imageData);
        System.out.println();
        MatrixUtils.printMatrix(rotatedImageData);

        System.out.println("Before rotation");
        System.out.println();
        System.out.println(predictor.predictDetailed(imageData));
        System.out.println();
        System.out.println(predictor.predictDetailed(rotatedImageData));

        int layerIndex = 0;
        double[][] origLayerWeights = weightsManager.getLayerWeights(layerIndex);
        double[][] rotatedWeights;
        rotatedWeights = AffineTransformations.rotate(origLayerWeights, 15, 16, rotationDegr);
        weightsManager.setLayerWeights(layerIndex, rotatedWeights);

        // double[][][] origLayerWeights = weightsManager.getAllWeights();
        // double[][] rotatedWeights;
        // for (int i = 0; i < origLayerWeights.length; i++) {
        //     rotatedWeights = AffineTransformations.rotate(origLayerWeights[i], 4, 5, rotationDegr);
        //     weightsManager.setLayerWeights(i, rotatedWeights);
        // }

        System.out.println("After rotation on " + rotationDegr);
        System.out.println();
        System.out.println(predictor.predictDetailed(imageData));
        System.out.println();
        System.out.println(predictor.predictDetailed(rotatedImageData));
    }
}
