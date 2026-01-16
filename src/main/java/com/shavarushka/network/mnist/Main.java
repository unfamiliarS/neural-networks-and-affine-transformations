package com.shavarushka.network.mnist;

import java.io.File;
import java.io.IOException;

import com.shavarushka.affine.MatrixUtils;
import com.shavarushka.affine.ScaleAffineTransformation;
import com.shavarushka.affine.ShearAffineTransformation;
import com.shavarushka.network.api.ModelLoader;
import com.shavarushka.network.api.NeuronActivationHandler;
import com.shavarushka.network.api.WeightsManager;
import com.shavarushka.network.api.fabric.MNISTModelFabric;
import com.shavarushka.network.api.fabric.ModelFabric;

public class Main {
    public static void main(String[] args) throws IOException {
        ModelFabric fabric = new MNISTModelFabric(ModelLoader.load("src/main/resources/simple-mnist/simple-mnist.zip"));

        WeightsManager weightsManager = fabric.createWeightsManager();
        NeuronActivationHandler neuronActivationHandler = fabric.createNeuronActivationHander();
        MNISTPredictor predictor = (MNISTPredictor) fabric.createPredictor();

        double rotationDegr = 256;
        ShearAffineTransformation affineTransformation = new ShearAffineTransformation()
                                                    .shear(0.02);

        double[][] imageData = ImageHandler.load(new File("src/main/resources/6_000445.png"));
        double[][] flattenImageData = new double[][]{ImageHandler.flattenImage(imageData)};
        affineTransformation.setMatrixType(true);
        double[][] rotatedImageData = affineTransformation.transform(flattenImageData);

        MatrixUtils.printMatrix(flattenImageData);
        System.out.println();
        MatrixUtils.printMatrix(rotatedImageData);

        System.out.println();
        System.out.println("Before weight rotation");
        System.out.println();
        System.out.println("Orig image");
        System.out.println("Neuron activations:");
        MatrixUtils.printMatrix(neuronActivationHandler.getAllLayerActivationsAsArrays(flattenImageData[0]));
        System.out.println();
        System.out.println(predictor.predict(flattenImageData[0]));
        System.out.println();
        System.out.println("Rotated image");
        System.out.println("Neuron activations:");
        MatrixUtils.printMatrix(neuronActivationHandler.getAllLayerActivationsAsArrays(rotatedImageData[0]));
        System.out.println();
        System.out.println(predictor.predict(rotatedImageData[0]));

        int layerIndex = 0;
        double[][] origLayerWeights = weightsManager.getLayerWeights(layerIndex);
        double[][] rotatedWeights;
        affineTransformation.setMatrixType(false);
        rotatedWeights = affineTransformation.transform(origLayerWeights);
        weightsManager.setLayerWeights(layerIndex, rotatedWeights);

        System.out.println("After weight rotation on " + rotationDegr);
        System.out.println();
        System.out.println("Orig image");
        System.out.println("Neuron activations:");
        MatrixUtils.printMatrix(neuronActivationHandler.getAllLayerActivationsAsArrays(flattenImageData[0]));
        System.out.println();
        System.out.println(predictor.predict(flattenImageData[0]));
        System.out.println();
        System.out.println("Rotated image");
        System.out.println("Neuron activations:");
        MatrixUtils.printMatrix(neuronActivationHandler.getAllLayerActivationsAsArrays(rotatedImageData[0]));
        System.out.println();
        System.out.println(predictor.predict(rotatedImageData[0]));
    }
}
