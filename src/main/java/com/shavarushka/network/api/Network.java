package com.shavarushka.network.api;

import java.io.IOException;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

public interface Network extends Savier {
    MultiLayerNetwork getModel();
    String getSummary();
    int getNumLayers();

    void save(String filePath) throws IOException;
}
