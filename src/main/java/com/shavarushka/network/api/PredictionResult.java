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

    @Override
    public String toString() {
        StringBuffer sb = new StringBuffer();
        sb.append("Predicted digit: " + predictedDigit + "\n");
        sb.append(String.format("Confidence: %.4f%n", confidence));
        sb.append("All probabilities:\n");
        for (int i = 0; i < probabilities.length; i++) {
            sb.append(String.format("  %d: %.4f%n", i, probabilities[i]));
        }

        return sb.toString();
    }
}