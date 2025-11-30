package com.shavarushka.network.multipletriangle;

import java.util.Arrays;

import com.shavarushka.affine.AffineTransformation;
import com.shavarushka.affine.MatrixUtils;
import com.shavarushka.affine.ShearAffineTransformation;
import com.shavarushka.network.api.ModelLoader;
import com.shavarushka.network.api.ModelPredictor;
import com.shavarushka.network.api.NeuronActivationHandler;
import com.shavarushka.network.api.WeightsManager;
import com.shavarushka.network.api.fabric.ModelFabric;
import com.shavarushka.network.api.fabric.MultipleTriangleModelFabric;

public class Main {
    public static void main(String[] args) {
        ModelFabric fabric = new MultipleTriangleModelFabric(ModelLoader.load("src/main/resources/two-triangle.zip"));

        ModelPredictor predictor = fabric.createPredictor();
        WeightsManager weightsManager = fabric.createWeightsManager();
        NeuronActivationHandler neuronActivationHandler = fabric.createNeuronActivationHander();

        double rotationDegr = 256;
        ShearAffineTransformation affineTransformation = new ShearAffineTransformation()
                                                        .shear(24);

        double[][] dataset = MultipleTriangleDataGenerator.getFromCSV("src/main/python/multipletriangle/dataset.csv");
        affineTransformation.setMatrixType(true);
        double[][] rotatedDataSet = affineTransformation.transform(dataset);
        int dataSetSampleIndex = 3;
        System.out.println("Data sample: " + Arrays.toString(dataset[dataSetSampleIndex]) + "\n");
        double[] dataSetSample = dataset[dataSetSampleIndex];
        double[] rotatedDataSetSample = rotatedDataSet[dataSetSampleIndex];
        
        System.out.println("Before weight rotation");
        System.out.println();
        System.out.println("Neuron activations:");
        MatrixUtils.printMatrix(neuronActivationHandler.getAllLayerActivationsAsArrays(dataSetSample));
        System.out.println();
        System.out.println(predictor.predict(dataSetSample));
        System.out.println();
        System.out.println("Neuron activations:");
        MatrixUtils.printMatrix(neuronActivationHandler.getAllLayerActivationsAsArrays(rotatedDataSetSample));
        System.out.println();
        System.out.println(predictor.predict(rotatedDataSetSample));

        System.out.println();

        for (int i = 0; i < 20; i++)
            System.out.println(Arrays.toString(dataset[i]));

        System.out.println();

        for (int i = 0; i < 20; i++)
            System.out.println(Arrays.toString(rotatedDataSet[i]));

        double[][][] allWeights = weightsManager.getAllWeights();
        affineTransformation.setMatrixType(false);
        double[][] rotatedWeights = affineTransformation.transform(allWeights[0]);
        weightsManager.setLayerWeights(0, rotatedWeights);

        System.out.println();
        System.out.println("After weight rotation on " + rotationDegr);
        System.out.println();
        System.out.println("Neuron activations:");
        MatrixUtils.printMatrix(neuronActivationHandler.getAllLayerActivationsAsArrays(dataSetSample));
        System.out.println();
        System.out.println(predictor.predict(dataSetSample));
        System.out.println();
        System.out.println("Neuron activations:");
        MatrixUtils.printMatrix(neuronActivationHandler.getAllLayerActivationsAsArrays(rotatedDataSetSample));
        System.out.println();
        System.out.println(predictor.predict(rotatedDataSetSample));
    }
}
