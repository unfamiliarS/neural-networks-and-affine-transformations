package com.shavarushka.network.multipletriangle;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import com.shavarushka.network.api.ModelPredictor;
import com.shavarushka.network.api.PredictionResult;

public class MultipleTrianglePredictor implements ModelPredictor {

    private MultiLayerNetwork model;

    public MultipleTrianglePredictor(MultiLayerNetwork model) {
        this.model = model;
    }

    @Override
    public PredictionResult predict(double[] input) {
        double x = input[0];
        double y = input[1];

        INDArray ndInput = Nd4j.create(new double[][]{{x, y}});
        INDArray output = model.output(ndInput);

        double[] probabilities = output.toDoubleVector();

        int predictedClass = -1;
        double maxProbability = -1.0;

        for (int i = 0; i < probabilities.length; i++) {
            if (probabilities[i] > maxProbability) {
                maxProbability = probabilities[i];
                predictedClass = i;
            }
        }

        String className = getClassName(predictedClass);

        return new MultipleTrianglePredictionResult(
            predictedClass,
            maxProbability,
            probabilities,
            className
        );
    }

    private String getClassName(int classIndex) {
        if (classIndex == 0) {
            return "Outside all triangles";
        } else {
            return "Inside triangle " + classIndex;
        }
    }
}
