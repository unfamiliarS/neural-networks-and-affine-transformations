package com.shavarushka.network.api;

public class PredictionResult {
    
    private final int predictedDigit;
    private final double confidence;
    private final double[] probabilities;

    public PredictionResult(int predictedDigit, double confidence, double[] probabilities) {
        this.predictedDigit = predictedDigit;
        this.confidence = confidence;
        this.probabilities = probabilities.clone();
    }

    public int getPredictedDigit() {
        return predictedDigit;
    }

    public double getConfidence() {
        return confidence;
    }

    public double[] getProbabilities() {
        return probabilities.clone();
    }

    public void printDetails() {
        System.out.println("Predicted digit: " + predictedDigit);
        System.out.printf("Confidence: %.4f%n", confidence);
        System.out.println("All probabilities:");
        for (int i = 0; i < probabilities.length; i++) {
            System.out.printf("  %d: %.4f%n", i, probabilities[i]);
        }
    }
}