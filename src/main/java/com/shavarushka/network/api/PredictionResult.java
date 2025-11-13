package com.shavarushka.network.api;

public class PredictionResult {

    protected int predictedDigit;
    protected double confidence;
    protected double[] probabilities;

    public PredictionResult(int predictedDigit, double confidence, double[] probabilities) {
        this.predictedDigit = predictedDigit;
        this.confidence = confidence;
        this.probabilities = probabilities.clone();
    }

    public PredictionResult(double confidence) {
        this.confidence = confidence;
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
        sb.append(String.format("Confidence: %.4f%n", confidence));

        return sb.toString();
    }
}