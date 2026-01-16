package com.shavarushka.network.api;

import java.util.List;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class NeuronActivationHandler {

    private MultiLayerNetwork network;

    public NeuronActivationHandler(MultiLayerNetwork network) {
        this.network = network;
    }

    public double[] getLayerActivationsAsArray(int layerIndex, double[] inputVector) {
        INDArray activations = getLayerActivations(layerIndex, inputVector);
        return activations.toDoubleVector();
    }

    public double[][] getAllLayerActivationsAsArrays(double[] inputVector) {
        List<INDArray> activationsArrays = getAllLayerActivations(inputVector);
        double[][] result = new double[activationsArrays.size()][];

        for (int i = 0; i < activationsArrays.size(); i++)
            result[i] = activationsArrays.get(i).toDoubleVector();

        return result;
    }

    public INDArray getLayerActivations(int layerIndex, double[] inputVector) {
        List<INDArray> activations = getAllLayerActivations(inputVector);
        if (layerIndex < 0 || layerIndex >= activations.size()) {
            throw new IllegalArgumentException("Invalid layer index: " + layerIndex +
                    ". Network has " + activations.size() + " layers.");
        }
        return activations.get(layerIndex);
    }

    public List<INDArray> getAllLayerActivations(double[] inputVector) {
        INDArray input = Nd4j.create(inputVector).reshape(1, inputVector.length);
        return network.feedForward(input);
    }
}
