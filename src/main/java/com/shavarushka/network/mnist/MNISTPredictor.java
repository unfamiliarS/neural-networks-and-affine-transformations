package com.shavarushka.network.mnist;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import com.shavarushka.network.api.ModelImagePredictor;
import com.shavarushka.network.api.PredictionResult;

import java.io.File;

public class MNISTPredictor implements ModelImagePredictor {

    private MultiLayerNetwork model;

    public MNISTPredictor(MultiLayerNetwork model) {
        this.model = model;
    }

    public PredictionResult predict(double[] imageData) {
        double[] probabilities = predictProbabilities(imageData);
        int predictedDigit = argMax(probabilities);
        double confidence = probabilities[predictedDigit];

        return new MNISTPredictionResult(predictedDigit, confidence, probabilities);
    }

    public PredictionResult predictFromImage(File imageFile) {
        double[] flattenImageData = ImageHandler.flattenImage(ImageHandler.load(imageFile));
        return predict(flattenImageData);
    }

    private double[] predictProbabilities(double[] imageData) {
        INDArray input = Nd4j.create(imageData, new int[]{1, 784});
        INDArray output = model.output(input);
        return output.toDoubleVector();
    }

    private int argMax(double[] array) {
        int maxIndex = 0;
        double maxValue = array[0];

        for (int i = 1; i < array.length; i++) {
            if (array[i] > maxValue) {
                maxValue = array[i];
                maxIndex = i;
            }
        }
        return maxIndex;
    }
}
