package com.shavarushka.network.mnist;

import java.io.File;
import java.io.IOException;

import com.shavarushka.affine.AffineTransformation;
import com.shavarushka.affine.MatrixUtils;
import com.shavarushka.affine.RotationMatrixProvider;
import com.shavarushka.network.api.ModelLoader;
import com.shavarushka.network.api.NeuronActivationHandler;
import com.shavarushka.network.api.WeightsManager;
import com.shavarushka.network.api.fabric.MNISTModelFabric;
import com.shavarushka.network.api.fabric.ModelFabric;

public class Main {
    public static void main(String[] args) throws IOException {
        ModelFabric fabric = new MNISTModelFabric(ModelLoader.load("src/main/resources/mnist-model.zip"));

        WeightsManager weightsManager = fabric.createWeightsManager();
        MNISTPredictor predictor = (MNISTPredictor) fabric.createPredictor();
        
        double rotationDegr = 256;
        AffineTransformation affineTransformation = new AffineTransformation(new RotationMatrixProvider()
                                                                            .setAngle(rotationDegr));

        double[][] imageData = ImageHandler.load(new File("src/main/resources/mnist-nums/8_005839.png"));
        double[][] flattenImageData = new double[][]{ImageHandler.flattenImage(imageData)};
        double[][] rotatedImageData = affineTransformation.transformComplex(flattenImageData);

        MatrixUtils.printMatrix(flattenImageData);
        System.out.println();
        MatrixUtils.printMatrix(rotatedImageData);

        System.out.println();
        System.out.println("Before weight rotation");
        System.out.println();
        System.out.println("Orig image");
        System.out.println("Neuron activations:");
        MatrixUtils.printMatrix(NeuronActivationHandler.getAllLayerActivationsAsArrays(fabric.createNetwork(), flattenImageData[0]));
        System.out.println();
        System.out.println(predictor.predict(flattenImageData[0]));
        System.out.println();
        System.out.println("Rotated image");
        System.out.println("Neuron activations:");
        MatrixUtils.printMatrix(NeuronActivationHandler.getAllLayerActivationsAsArrays(fabric.createNetwork(), rotatedImageData[0]));
        System.out.println();
        System.out.println(predictor.predict(rotatedImageData[0]));

        int layerIndex = 0;
        double[][] origLayerWeights = weightsManager.getLayerWeights(layerIndex);
        double[][] rotatedWeights;
        rotatedWeights = affineTransformation.transformComplex(origLayerWeights);
        weightsManager.setLayerWeights(layerIndex, rotatedWeights);

        System.out.println("After weight rotation on " + rotationDegr);
        System.out.println();
        System.out.println("Orig image");
        System.out.println("Neuron activations:");
        MatrixUtils.printMatrix(NeuronActivationHandler.getAllLayerActivationsAsArrays(fabric.createNetwork(), flattenImageData[0]));
        System.out.println();
        System.out.println(predictor.predict(flattenImageData[0]));
        System.out.println();
        System.out.println("Rotated image");
        System.out.println("Neuron activations:");
        MatrixUtils.printMatrix(NeuronActivationHandler.getAllLayerActivationsAsArrays(fabric.createNetwork(), rotatedImageData[0]));
        System.out.println();
        System.out.println(predictor.predict(rotatedImageData[0]));
    }
}
