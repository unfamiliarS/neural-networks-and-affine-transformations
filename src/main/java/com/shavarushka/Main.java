package com.shavarushka;

import java.io.File;
import java.io.IOException;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

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
        int axis1 = 3, axis2 = 10;

        double[][] imageData = predictor.load(new File("src/main/resources/mnist-nums/8_005839.png"));
        double[][] rotatedImageData = AffineTransformations.strictRotate(imageData, axis1, axis2, rotationDegr);

        MatrixUtils.printMatrix(imageData);
        System.out.println();
        MatrixUtils.printMatrix(rotatedImageData);

        System.out.println();
        System.out.println("Before weight rotation");
        System.out.println();
        System.out.println("Orig image");
        System.out.println(predictor.predictDetailed(imageData));
        System.out.println();
        System.out.println("Rotated image");
        System.out.println(predictor.predictDetailed(rotatedImageData));

        int layerIndex = 0;
        double[][] origLayerWeights = weightsManager.getLayerWeights(layerIndex);
        double[][] rotatedWeights;
        rotatedWeights = AffineTransformations.strictRotate(origLayerWeights, axis1, axis2, rotationDegr);
        weightsManager.setLayerWeights(layerIndex, rotatedWeights);

        // double[][][] origLayerWeights = weightsManager.getAllWeights();
        // double[][] rotatedWeights;
        // for (int i = 0; i < origLayerWeights.length; i++) {
        //     rotatedWeights = AffineTransformations.strictRotate(origLayerWeights[i], axis1, axis2, rotationDegr);
        //     weightsManager.setLayerWeights(i, rotatedWeights);
        // }

        System.out.println("After weight rotation on " + rotationDegr);
        System.out.println();
        System.out.println("Orig image");
        System.out.println(predictor.predictDetailed(imageData));
        System.out.println();
        System.out.println("Rotated image");
        System.out.println(predictor.predictDetailed(rotatedImageData));

    }
}
