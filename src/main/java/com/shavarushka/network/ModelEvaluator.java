package com.shavarushka.network;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import com.shavarushka.network.api.DataIterators;

import org.nd4j.evaluation.classification.Evaluation;

public class ModelEvaluator {
    private MultiLayerNetwork model;
    private DataIterators dataIterators;

    public ModelEvaluator(MultiLayerNetwork model, DataIterators dataIterators) {
        this.model = model;
        this.dataIterators = dataIterators;
    }

    public Evaluation evaluate() {
        DataSetIterator testIterator = dataIterators.getTestIterator();
        Evaluation evaluation = model.evaluate(testIterator);
        dataIterators.resetTestIterator();
        return evaluation;
    }

    public void printEvaluation() {
        Evaluation evaluation = evaluate();
        System.out.println(evaluation.stats());
        System.out.println("Confusion Matrix:\n" + evaluation.confusionToString());
    }

    public double calculateAccuracy() {
        Evaluation evaluation = evaluate();
        return evaluation.accuracy();
    }

    public void printAccuracy() {
        double accuracy = calculateAccuracy();
        System.out.println("Accuracy: " + String.format("%.4f", accuracy));
    }
}
