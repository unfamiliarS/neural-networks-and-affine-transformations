package com.shavarushka.network.api.fabric;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

public class ModelFactoryOfFactory {

    private static ModelFabric instance;

    public static ModelFabric createFabric(String fabricType, MultiLayerNetwork net) {
        if (instance == null) {
            synchronized (ModelFactoryOfFactory.class) {
                if (instance == null) {
                    instance = switch (fabricType.toLowerCase()) {
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
        }

        return instance;   
    }
}
