package com.shavarushka.network.mnist;

import java.io.File;
import java.io.IOException;

import com.shavarushka.affine.AffineTransformations;
import com.shavarushka.affine.MatrixUtils;
import com.shavarushka.network.api.ModelLoader;
import com.shavarushka.network.api.WeightsManager;
import com.shavarushka.network.api.fabric.MNISTModelFabric;
import com.shavarushka.network.api.fabric.ModelFabric;

public class Main {
    public static void main(String[] args) throws IOException {
        ModelFabric fabric = new MNISTModelFabric(ModelLoader.load("simple-mnist.zip"));

        WeightsManager weightsManager = fabric.createWeightsManager();
        MNISTPredictor predictor = (MNISTPredictor) fabric.createPredictor();

        double rotationDegr = 256;
        int axis1 = 180, axis2 = 181;

        double[][] imageData = ImageHandler.load(new File("src/main/resources/mnist-nums/2_000587.png"));
        double[][] flattenImageData = new double[][]{ImageHandler.flattenImage(imageData)};
        double[][] rotatedImageData = AffineTransformations.rotateComplex(flattenImageData, rotationDegr);

        MatrixUtils.printMatrix(flattenImageData);
        System.out.println();
        MatrixUtils.printMatrix(rotatedImageData);

        System.out.println();
        System.out.println("Before weight rotation");
        System.out.println();
        System.out.println("Orig image");
        System.out.println(predictor.predict(flattenImageData[0]));
        System.out.println();
        System.out.println("Rotated image");
        System.out.println(predictor.predict(rotatedImageData[0]));

        int layerIndex = 0;
        double[][] origLayerWeights = weightsManager.getLayerWeights(layerIndex);
        double[][] rotatedWeights;
        rotatedWeights = AffineTransformations.rotateComplex(origLayerWeights, rotationDegr);
        weightsManager.setLayerWeights(layerIndex, rotatedWeights);

        System.out.println("After weight rotation on " + rotationDegr);
        System.out.println();
        System.out.println("Orig image");
        System.out.println(predictor.predict(flattenImageData[0]));
        System.out.println();
        System.out.println("Rotated image");
        System.out.println(predictor.predict(rotatedImageData[0]));
    }
}
