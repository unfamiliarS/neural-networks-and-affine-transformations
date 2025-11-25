package com.shavarushka.network.api.fabric;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

import com.shavarushka.network.api.ModelEvaluator;
import com.shavarushka.network.api.ModelPredictor;
import com.shavarushka.network.api.ModelTrainer;
import com.shavarushka.network.api.WeightsManager;

public abstract class ModelFabric {

    protected MultiLayerNetwork network;

    protected ModelFabric(MultiLayerNetwork net) {
        network = net;
    }

    public abstract MultiLayerNetwork createNetwork();
    public abstract ModelTrainer createTrainer();
    public abstract ModelEvaluator createEvaluator();
    public abstract ModelPredictor createPredictor();

    public WeightsManager createWeightsManager() {
        return new WeightsManager(network);
    }

    public static ModelFabric createFabric(String fabricType, MultiLayerNetwork net) {
        return switch (fabricType.toLowerCase()) {
            case "mnist" -> new MNISTModelFabric(net);
            case "triangle" -> new TriangleModelFabric(net);
            case "multipletriangle" -> new MultipleTriangleModelFabric(net);
            default -> throw new IllegalArgumentException(
                String.format("Unknown model type: '%s'. Supported types: %s", 
                    fabricType, 
                    "mnist, triangle, multipletriangle")
            );
        };
    }
}
