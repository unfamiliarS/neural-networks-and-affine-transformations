package com.shavarushka.network.api.fabric;

import com.shavarushka.network.api.ModelLoader;
import static com.shavarushka.network.api.Models.*;

public class ModelFactoryOfFactory {

    private static ModelFabric instance;

    public static ModelFabric createFabric(String fabricType) {
        if (instance == null) {
            synchronized (ModelFactoryOfFactory.class) {
                if (instance == null) {
                    instance = switch (fabricType.toLowerCase()) {
                        case "mnist" -> new MNISTModelFabric(ModelLoader.load(MNIST.getModelPath()));
                        case "simple-mnist" -> new MNISTModelFabric(ModelLoader.load(SIMPLE_MNIST.getModelPath()));
                        case "triangle" -> new TriangleModelFabric(ModelLoader.load(TRIANGLE.getModelPath()));
                        case "two-triangles" -> new MultipleTriangleModelFabric(ModelLoader.load(TWO_TRIANGLES.getModelPath()));
                        default -> throw new IllegalArgumentException(
                            String.format("Unknown model type: '%s'. Supported types: %s",
                                fabricType,
                                "mnist, triangle, two-triangles")
                        );
                    };
                }
            }
        }

        return instance;
    }
}
