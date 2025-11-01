package com.shavarushka.network.api;

import java.io.File;
import java.io.IOException;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;

public class ModelSavier {

    public static void save(MultiLayerNetwork model, String filePath) {
        try {
            ModelSerializer.writeModel(model, new File(filePath), true);
            System.out.println("Model saved: " + filePath);
            System.out.println();
        } catch (IOException e) {
            System.err.println("Failed to save model: " + e.getMessage());
        }
    }
}
