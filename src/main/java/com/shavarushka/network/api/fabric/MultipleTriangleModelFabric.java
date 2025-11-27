package com.shavarushka.network.api.fabric;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

import com.shavarushka.network.api.DataGenerator;
import com.shavarushka.network.api.GeneratedDataEvaluator;
import com.shavarushka.network.api.GeneratedDataTrainer;
import com.shavarushka.network.api.ModelEvaluator;
import com.shavarushka.network.api.ModelPredictor;
import com.shavarushka.network.api.ModelTrainer;
import com.shavarushka.network.multipletriangle.MultipleTriangleDataGenerator;
import com.shavarushka.network.multipletriangle.MultipleTriangleNetwork;
import com.shavarushka.network.triangle.TrianglePredictor;

public class MultipleTriangleModelFabric extends ModelFabric {

    private DataGenerator testDataGenerator = new MultipleTriangleDataGenerator(1000, false);
    private ModelEvaluator evaluator = new GeneratedDataEvaluator(network, testDataGenerator);

    public MultipleTriangleModelFabric() {
        super(MultipleTriangleNetwork.create());
    }

    public MultipleTriangleModelFabric(MultiLayerNetwork net) {
        super(net);
    }

    @Override
    public MultiLayerNetwork createNetwork() {
        return network;
    }

    @Override
    public ModelTrainer createTrainer() {
        DataGenerator trainDataGenerator = new MultipleTriangleDataGenerator(1000, true);
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
