package com.shavarushka.network.api;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

public abstract class ModelTrainer implements Trainer {

    protected MultiLayerNetwork model;
    protected ModelEvaluator evaluator;

    public ModelTrainer(MultiLayerNetwork model, ModelEvaluator evaluator) {
        this.model = model;
        this.evaluator = evaluator;
    }
}
