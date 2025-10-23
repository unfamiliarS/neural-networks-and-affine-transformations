package com.shavarushka.network.api;

public interface Classifier extends Savier {
    void train(int numEpochs);
    void evaluate();
    // void predict()
}
