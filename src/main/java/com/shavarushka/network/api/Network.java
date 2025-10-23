package com.shavarushka.network.api;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

public interface Network extends Savier {
    MultiLayerNetwork getModel();
}
