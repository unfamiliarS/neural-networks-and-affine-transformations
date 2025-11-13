package com.shavarushka.network.triangle;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import com.shavarushka.network.api.ModelPredictor;
import com.shavarushka.network.api.PredictionResult;

public class TrianglePredictor implements ModelPredictor {

    private MultiLayerNetwork model;

    public TrianglePredictor(MultiLayerNetwork model) {
        this.model = model;
    }

    @Override
    public PredictionResult predict(Object input) {
        double[] coordinates = (double[]) input;
        double x = coordinates[0];
        double y = coordinates[1];
        
        INDArray ndInput = Nd4j.create(new double[][]{{x, y}});
        INDArray output = model.output(ndInput);
        double confidence = output.getDouble(0);
        
        return new PredictionResult(confidence);
    }
}
