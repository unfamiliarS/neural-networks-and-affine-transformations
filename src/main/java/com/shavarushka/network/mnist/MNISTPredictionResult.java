package com.shavarushka.network.mnist;

import com.shavarushka.network.api.PredictionResult;

public class MNISTPredictionResult extends PredictionResult {

    public MNISTPredictionResult(int predictedDigit, double confidence, double[] probabilities) {
        super(predictedDigit, confidence, probabilities);
    }

    @Override
    public String toString() {
        StringBuffer sb = new StringBuffer();
        sb.append("Predicted digit: " + predictedDigit + "\n");
        sb.append("Confidence: " + confidence + "\n");
        sb.append("All probabilities:\n");
        for (int i = 0; i < probabilities.length; i++)
            sb.append("  " + i + ": " + probabilities[i] + "\n");

        return sb.toString();
    }
}
