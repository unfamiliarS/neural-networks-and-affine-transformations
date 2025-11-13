package com.shavarushka.network.api.fabric;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

import com.shavarushka.network.api.DataGenerator;
import com.shavarushka.network.api.GeneratedDataEvaluator;
import com.shavarushka.network.api.GeneratedDataTrainer;
import com.shavarushka.network.api.ModelEvaluator;
import com.shavarushka.network.api.ModelPredictor;
import com.shavarushka.network.api.ModelTrainer;
import com.shavarushka.network.triangle.TriangleDataGenerator;
import com.shavarushka.network.triangle.TriangleNetwork;
import com.shavarushka.network.triangle.TrianglePredictor;

public class TriangleModelFabric extends ModelFabric {

    private DataGenerator testDataGenerator = new TriangleDataGenerator(1000, false);
    private ModelEvaluator evaluator = new GeneratedDataEvaluator(network, testDataGenerator);

    public TriangleModelFabric() {
        super(TriangleNetwork.create());
    }

    public TriangleModelFabric(MultiLayerNetwork net) {
        super(net);
    }

    @Override
    public MultiLayerNetwork createNetwork() {
        return network;
    }

    @Override
    public ModelTrainer createTrainer() {
        DataGenerator trainDataGenerator = new TriangleDataGenerator(1000, true);
        return new GeneratedDataTrainer(network, trainDataGenerator, evaluator);
    }

    @Override
    public ModelEvaluator createEvaluator() {
        return evaluator;
    }

    @Override
    public ModelPredictor createPredictor() {
        return new TrianglePredictor(network);
    }
}
