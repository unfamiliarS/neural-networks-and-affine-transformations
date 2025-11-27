package com.shavarushka.network.api;

import java.util.List;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class NeuronActivationHandler {

    public static double[] getLayerActivationsAsArray(MultiLayerNetwork network, int layerIndex, double[] inputVector) {
        INDArray activations = getLayerActivations(network, layerIndex, inputVector);
        return activations.toDoubleVector();
    }

    public static double[][] getAllLayerActivationsAsArrays(MultiLayerNetwork network, double[] inputVector) {
        List<INDArray> activationsArrays = getAllLayerActivations(network, inputVector);
        double[][] result = new double[activationsArrays.size()][];
        
        for (int i = 0; i < activationsArrays.size(); i++)
            result[i] = activationsArrays.get(i).toDoubleVector();
        
        return result;
    }

    public static INDArray getLayerActivations(MultiLayerNetwork network, int layerIndex, double[] inputVector) {
        List<INDArray> activations = getAllLayerActivations(network, inputVector);
        if (layerIndex < 0 || layerIndex >= activations.size()) {
            throw new IllegalArgumentException("Invalid layer index: " + layerIndex + 
                    ". Network has " + activations.size() + " layers.");
        }
        return activations.get(layerIndex);
    }

    public static List<INDArray> getAllLayerActivations(MultiLayerNetwork network, double[] inputVector) {
        INDArray input = Nd4j.create(inputVector).reshape(1, inputVector.length);
        return network.feedForward(input);
    }
}
