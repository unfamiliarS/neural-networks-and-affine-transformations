package com.shavarushka.network.api;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.evaluation.classification.Evaluation;

public abstract class ModelEvaluator implements Evaluator {

    protected MultiLayerNetwork model;

    public ModelEvaluator(MultiLayerNetwork model) {
        this.model = model;
    }

    public void printEvaluation() {
        Evaluation evaluation = evaluate();
        System.out.println(evaluation.stats());
        System.out.println("Confusion Matrix:\n" + evaluation.confusionToString());
    }

    public void printAccuracy() {
        double accuracy = calculateAccuracy();
        System.out.println("Accuracy: " + String.format("%.4f", accuracy));
    }
    
    public double calculateAccuracy() {
        Evaluation evaluation = evaluate();
        return evaluation.accuracy();
    }
}
