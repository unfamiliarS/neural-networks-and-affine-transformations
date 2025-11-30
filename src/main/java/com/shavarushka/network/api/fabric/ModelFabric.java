package com.shavarushka.network.api.fabric;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

import com.shavarushka.network.api.ModelEvaluator;
import com.shavarushka.network.api.ModelPredictor;
import com.shavarushka.network.api.ModelTrainer;
import com.shavarushka.network.api.NeuronActivationHandler;
import com.shavarushka.network.api.WeightsManager;

public abstract class ModelFabric {

    protected MultiLayerNetwork network;

    protected ModelFabric(MultiLayerNetwork net) {
        network = net;
    }

    public abstract MultiLayerNetwork createNetwork();
    public abstract ModelTrainer createTrainer();
    public abstract ModelEvaluator createEvaluator();
    public abstract ModelPredictor createPredictor();

    public WeightsManager createWeightsManager() {
        return new WeightsManager(network);
    }

    public NeuronActivationHandler createNeuronActivationHander() {
        return new NeuronActivationHandler(network);
    }
}
