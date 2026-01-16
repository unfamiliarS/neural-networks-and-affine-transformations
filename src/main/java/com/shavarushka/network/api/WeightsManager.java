package com.shavarushka.network.api;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;

public class WeightsManager {

    private MultiLayerNetwork model;

    public WeightsManager(MultiLayerNetwork model) {
        this.model = model;
    }

    public double[][] getLayerWeights(int layerIndex) {
        INDArray weights = model.getLayer(layerIndex).getParam("W");
        double[][] weightsArray = new double[weights.columns()][weights.rows()];
        for (int i = 0; i < weights.columns(); i++)
            for (int j = 0; j < weights.rows(); j++)
                weightsArray[i][j] = weights.getDouble(j, i);

        return weightsArray;
    }

    public double[][][] getAllWeights() {
        int numLayers = model.getnLayers();
        double[][][] weights = new double[numLayers][][];
        for (int i = 0; i < numLayers; i++)
            weights[i] = getLayerWeights(i);

        return weights;
    }

    public void setLayerWeights(int layerIndex, double[][] newWeights) {
        INDArray weights = model.getLayer(layerIndex).getParam("W");

        double[][] returnedWeights = new double[weights.rows()][weights.columns()];
        for (int i = 0; i < weights.rows(); i++)
            for (int j = 0; j < weights.columns(); j++)
                returnedWeights[i][j] = newWeights[j][i];

        INDArray newWeightsArray = Nd4j.create(returnedWeights);
        model.getLayer(layerIndex).setParam("W", newWeightsArray);
    }

    public void setAllWeights(double[][][] allWeights) {
        for (int i = 0; i < allWeights.length; i++) {
            setLayerWeights(i, allWeights[i]);
        }
    }

    public double[] getLayerBiases(int layerIndex) {
        INDArray biases = model.getLayer(layerIndex).getParam("b");
        double[] biasesArray = new double[(int) biases.length()];
        for (int i = 0; i < biases.length(); i++)
            biasesArray[i] = biases.getDouble(i);

        return biasesArray;
    }

    public double[][] getAllBiases() {
        int numLayers = model.getnLayers();
        double[][] biases = new double[numLayers][];
        for (int i = 0; i < numLayers; i++) {
            biases[i] = getLayerBiases(i);
        }
        return biases;
    }

    public void printWeights() {
        double[][][] matrix = getAllWeights();
        for (int i = 0; i < matrix.length; i++) {
            System.out.println("Layer " + i + ":");
            double[][] weightsArray = getLayerWeights(i);
            for (double[] row : weightsArray) {
                System.out.println(Arrays.toString(row));
            }
        }
    }

    public void printBiases() {
        double[][] biases = getAllBiases();
        for (int i = 0; i < biases.length; i++) {
            System.out.println("Layer " + i + ":");
            System.out.println(Arrays.toString(biases[i]));
        }
    }
}
