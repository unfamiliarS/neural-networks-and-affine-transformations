package com.shavarushka.network.mnist;

import java.io.File;

import com.shavarushka.network.api.ModelEvaluator;
import com.shavarushka.network.api.ModelSavier;
import com.shavarushka.network.api.ModelTrainer;
import com.shavarushka.network.api.fabric.MNISTModelFabric;
import com.shavarushka.network.api.fabric.ModelFabric;

public class TrainNew {
    public static void main(String[] args) {
        ModelFabric fabric = new MNISTModelFabric(MNISTExperimentalNetwork.create());
        ModelTrainer trainer = fabric.createTrainer();
        ModelEvaluator evaluator = fabric.createEvaluator();
        MNISTPredictor predictor = (MNISTPredictor) fabric.createPredictor();
        trainer.train(10);
        evaluator.printEvaluation();

        System.out.println(predictor.predictFromImage(new File("src/main/resources/mnist-nums/4_000673.png")));
        ModelSavier.save(fabric.getNetwork(), "simple-mnist.zip");
    }
}
