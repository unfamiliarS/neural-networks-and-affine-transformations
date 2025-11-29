package com.shavarushka.network.multipletriangle;

import java.util.Arrays;

import com.shavarushka.affine.AffineTransformation;
import com.shavarushka.affine.ScaleAffineTransformation;
import com.shavarushka.network.api.ModelLoader;
import com.shavarushka.network.api.ModelPredictor;
import com.shavarushka.network.api.WeightsManager;
import com.shavarushka.network.api.fabric.ModelFabric;
import com.shavarushka.network.api.fabric.MultipleTriangleModelFabric;

public class Main {
    public static void main(String[] args) {
        ModelFabric fabric = new MultipleTriangleModelFabric(ModelLoader.load("src/main/resources/two-triangles.zip"));

        ModelPredictor predictor = fabric.createPredictor();
        WeightsManager weightsManager = fabric.createWeightsManager();

        double rotationDegr = 256;
        AffineTransformation affineTransformation = new ScaleAffineTransformation()
                                                        .scaleFactor(5);

        System.out.println();
        System.out.println("Weights:");
        weightsManager.printWeights();
        System.out.println();
        System.out.println("Biases:");
        weightsManager.printBiases();
        System.out.println();

        double[][] dataset = MultipleTriangleDataGenerator.getFromCSV("src/main/python/multipletriangle/dataset.csv");
        ((ScaleAffineTransformation) affineTransformation).setMatrixType(true);
        double[][] rotatedDataSet = affineTransformation.transform(dataset);
        int dataSetSampleIndex = 3;
        System.out.println("Data sample: " + Arrays.toString(dataset[dataSetSampleIndex]) + "\n");
        double[] dataSetSample = dataset[dataSetSampleIndex];
        double[] rotatedDataSetSample = rotatedDataSet[dataSetSampleIndex];

        System.out.println("Before weight rotation");
        System.out.println();
        System.out.println(predictor.predict(dataSetSample));

        System.out.println();
        System.out.println(predictor.predict(rotatedDataSetSample));

        System.out.println();

        for (int i = 0; i < 20; i++)
            System.out.println(Arrays.toString(dataset[i]));

        System.out.println();

        for (int i = 0; i < 20; i++)
            System.out.println(Arrays.toString(rotatedDataSet[i]));

        double[][][] allWeights = weightsManager.getAllWeights();
        ((ScaleAffineTransformation) affineTransformation).setMatrixType(false);
        double[][] rotatedWeights = affineTransformation.transform(allWeights[0]);
        weightsManager.setLayerWeights(0, rotatedWeights);

        System.out.println();
        System.out.println("Weights:");
        weightsManager.printWeights();
        System.out.println();
        System.out.println("Biases:");
        weightsManager.printBiases();
        System.out.println();

        System.out.println();
        System.out.println("After weight rotation on " + rotationDegr);
        System.out.println();
        System.out.println(predictor.predict(dataSetSample));

        System.out.println();
        System.out.println(predictor.predict(rotatedDataSetSample));
    }
}
