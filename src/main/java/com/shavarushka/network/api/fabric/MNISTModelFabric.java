package com.shavarushka.network.api.fabric;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

import com.shavarushka.network.api.DataIterators;
import com.shavarushka.network.api.IteratorDataEvaluator;
import com.shavarushka.network.api.IteratorDataTrainer;
import com.shavarushka.network.api.ModelEvaluator;
import com.shavarushka.network.api.ModelPredictor;
import com.shavarushka.network.api.ModelTrainer;
import com.shavarushka.network.mnist.MNISTDataIterators;
import com.shavarushka.network.mnist.MNISTNetwork;
import com.shavarushka.network.mnist.MNISTPredictor;

public class MNISTModelFabric extends ModelFabric {

    private DataIterators dataIterators = new MNISTDataIterators();
    private ModelEvaluator evaluator = new IteratorDataEvaluator(network, dataIterators);

    public MNISTModelFabric() {
        super(MNISTNetwork.create());
    }

    public MNISTModelFabric(MultiLayerNetwork net) {
        super(net);
    }

    @Override
    public MultiLayerNetwork createNetwork() {
        return network;
    }

    @Override
    public ModelTrainer createTrainer() {
        return new IteratorDataTrainer(network, dataIterators, evaluator);
    }

    @Override
    public ModelEvaluator createEvaluator() {
        return evaluator;
    }

    @Override
    public ModelPredictor createPredictor() {
        return new MNISTPredictor(network);
    }
}
