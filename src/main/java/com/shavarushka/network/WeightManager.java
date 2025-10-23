package com.shavarushka.network;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;

public class WeightManager {
    
    private MultiLayerNetwork model;

    public WeightManager(MultiLayerNetwork model) {
        this.model = model;
    }

    public double[][] getLayerWeights(int layerIndex) {
        INDArray weights = model.getLayer(layerIndex).getParam("W");
        double[][] weightsArray = new double[weights.rows()][weights.columns()];
        for (int i = 0; i < weights.rows(); i++) {
            for (int j = 0; j < weights.columns(); j++) {
                weightsArray[i][j] = weights.getDouble(i, j);
            }
        }
        return weightsArray;
    }

    public double[][][] getAllWeights() {
        int numLayers = model.getnLayers();
        double[][][] weights = new double[numLayers][][];
        for (int i = 0; i < numLayers; i++) {
            weights[i] = getLayerWeights(i);
        }
        return weights;
    }

    public void setLayerWeights(int layerIndex, double[][] newWeights) {
        INDArray weights = model.getLayer(layerIndex).getParam("W");
        INDArray newWeightsArray = Nd4j.create(newWeights).reshape(weights.shape());
        model.getLayer(layerIndex).setParam("W", newWeightsArray);
    }

    public void setAllWeights(double[][][] allWeights) {
        for (int i = 0; i < allWeights.length; i++) {
            setLayerWeights(i, allWeights[i]);
        }
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

    public long getLayerParameterCount(int layerIndex) {
        return model.getLayer(layerIndex).numParams();
    }
}
