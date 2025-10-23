package com.shavarushka.network.api;

import java.io.File;
import java.io.IOException;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;

public abstract class BaseNetwork implements Network {
    
    protected MultiLayerNetwork model;

    public BaseNetwork(MultiLayerNetwork model) {
        this.model = model;
        this.model.init();
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
