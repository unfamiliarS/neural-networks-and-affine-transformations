package com.shavarushka.network.triangle;

import java.util.Arrays;

import com.shavarushka.affine.AffineTransformations;
import com.shavarushka.network.api.DataGenerator;
import com.shavarushka.network.api.ModelLoader;
import com.shavarushka.network.api.ModelPredictor;
import com.shavarushka.network.api.WeightsManager;
import com.shavarushka.network.api.fabric.ModelFabric;
import com.shavarushka.network.api.fabric.TriangleModelFabric;

public class Main {
    public static void main(String[] args) {
        ModelFabric fabric = new TriangleModelFabric(ModelLoader.load("src/main/resources/triangle-classifier.zip"));

        ModelPredictor predictor = fabric.createPredictor();
        WeightsManager weightsManager = fabric.createWeightsManager();
        System.out.println();
        weightsManager.printWeights();
        System.out.println();

        double rotationDegr = 180;

        double[][] dataset = DataGenerator.getFromCSV("src/main/python/triangle/dataset.csv");
        double[][] rotatedDataSet = AffineTransformations.rotate(dataset, 0, 1, rotationDegr);
        int dataSetSampleIndex = 6;
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
        double[][] rotatedWeights = AffineTransformations.rotate(allWeights[0], 0, 1, rotationDegr);
        weightsManager.setLayerWeights(0, rotatedWeights);

        System.out.println();
        System.out.println("After weight rotation on " + rotationDegr);
        System.out.println();
        System.out.println(predictor.predict(dataSetSample));

        System.out.println();
        System.out.println(predictor.predict(rotatedDataSetSample));
    }
}
