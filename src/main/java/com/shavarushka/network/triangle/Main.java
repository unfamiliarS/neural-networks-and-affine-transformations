package com.shavarushka.network.triangle;

import java.util.Arrays;

import com.shavarushka.affine.MatrixUtils;
import com.shavarushka.affine.ScaleAffineTransformation;
import com.shavarushka.network.api.ModelLoader;
import com.shavarushka.network.api.ModelPredictor;
import com.shavarushka.network.api.NeuronActivationHandler;
import com.shavarushka.network.api.WeightsManager;
import com.shavarushka.network.api.fabric.ModelFabric;
import com.shavarushka.network.api.fabric.TriangleModelFabric;

public class Main {
    public static void main(String[] args) {
        ModelFabric fabric = new TriangleModelFabric(ModelLoader.load("src/main/resources/triangle-classifier.zip"));

        ModelPredictor predictor = fabric.createPredictor();
        WeightsManager weightsManager = fabric.createWeightsManager();

        double rotationDegr = 180;
        ScaleAffineTransformation affineTransformation = new ScaleAffineTransformation()
                                                                .scaleFactor(5);

        System.out.println();
        weightsManager.printWeights();
        System.out.println();
        
        
        double[][] dataset = TriangleDataGenerator.getFromCSV("src/main/python/triangle/dataset.csv");
        affineTransformation.setMatrixType(true);
        double[][] rotatedDataSet = affineTransformation.transform(dataset);
        int dataSetSampleIndex = 6;
        double[] dataSetSample = dataset[dataSetSampleIndex];
        double[] rotatedDataSetSample = rotatedDataSet[dataSetSampleIndex];

        MatrixUtils.printMatrix(NeuronActivationHandler.getAllLayerActivationsAsArrays(fabric.createNetwork(), dataSetSample));
        
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
        affineTransformation.setMatrixType(false);
        double[][] rotatedWeights = affineTransformation.transform(allWeights[0]);
        weightsManager.setLayerWeights(0, rotatedWeights);

        System.out.println();
        weightsManager.printWeights();
        System.out.println();
        MatrixUtils.printMatrix(NeuronActivationHandler.getAllLayerActivationsAsArrays(fabric.createNetwork(), rotatedDataSetSample));

        System.out.println();
        System.out.println("After weight rotation on " + rotationDegr);
        System.out.println();
        System.out.println(predictor.predict(dataSetSample));

        System.out.println();
        System.out.println(predictor.predict(rotatedDataSetSample));
    }
}
