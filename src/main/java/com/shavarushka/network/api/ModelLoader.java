package com.shavarushka.network.api;

import java.io.File;
import java.io.IOException;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;

public class ModelLoader {

    public static MultiLayerNetwork load(String filePath) {
        try {
            MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(new File(filePath));
            System.out.println("Model loaded: " + filePath);
            return model;
        } catch (IOException e) {
            System.err.println("Failed to load model: " + e.getMessage());
            return null;
        }
    }
}
