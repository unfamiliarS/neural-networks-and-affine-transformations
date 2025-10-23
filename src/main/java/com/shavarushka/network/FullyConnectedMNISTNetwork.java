package com.shavarushka.network;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import com.shavarushka.network.api.Network;

import java.io.File;
import java.io.IOException;

public class FullyConnectedMNISTNetwork implements Network {

    private MultiLayerNetwork model;
    private static final int numClasses = 10;
    private static final int seed = 12345;

    public FullyConnectedMNISTNetwork(MultiLayerNetwork model) {
        this.model = model;
        this.model.init();
    }

    public static Network create() {
        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Adam(0.001))
                .l2(1e-4)
                .list()
                .layer(new DenseLayer.Builder()
                        .nIn(28 * 28)
                        .nOut(64)
                        .activation(Activation.RELU)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .layer(new DenseLayer.Builder()
                        .nIn(64)
                        .nOut(32)
                        .activation(Activation.RELU)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .layer(new DenseLayer.Builder()
                        .nIn(32)
                        .nOut(16)
                        .activation(Activation.RELU)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nIn(16)
                        .nOut(numClasses)
                        .activation(Activation.SOFTMAX)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(config);
        return new FullyConnectedMNISTNetwork(model);
    }

    public static FullyConnectedMNISTNetwork load(String filePath) throws IOException {
        MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(new File(filePath));
        System.out.println("Model loaded: " + filePath);
        return new FullyConnectedMNISTNetwork(model);
    }

    public void save(String filePath) throws IOException {
        ModelSerializer.writeModel(model, new File(filePath), true);
        System.out.println("Model saved: " + filePath);
    }

    public MultiLayerNetwork getModel() {
        return model;
    }

    public String getSummary() {
        return model.summary();
    }

    public int getNumLayers() {
        return model.getnLayers();
    }
}
