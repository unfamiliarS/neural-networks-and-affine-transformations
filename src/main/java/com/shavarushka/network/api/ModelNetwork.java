package com.shavarushka.network.api;

import java.io.File;
import java.io.IOException;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;

public class ModelNetwork implements Savier {

    protected MultiLayerNetwork model;

    protected ModelNetwork(MultiLayerNetwork model) {
        this.model = model;
        this.model.init();
    }

    public static ModelNetwork getFrom(MultiLayerNetwork model) {
        return new ModelNetwork(model);
    }

    public static ModelNetwork load(String filePath) {
        try {
            MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(new File(filePath));
            System.out.println("Model loaded: " + filePath);
            return new ModelNetwork(model);
        } catch (IOException e) {
            System.err.println("Failed to load model: " + e.getMessage());
            return null;
        }
    }

    public void save(String filePath) throws IOException {
        ModelSerializer.writeModel(model, new File(filePath), true);
        System.out.println("Model saved: " + filePath);
        System.out.println();
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
