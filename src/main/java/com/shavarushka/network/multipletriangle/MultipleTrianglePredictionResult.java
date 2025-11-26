package com.shavarushka.network.multipletriangle;

import com.shavarushka.network.api.PredictionResult;

public class MultipleTrianglePredictionResult extends PredictionResult {
    private final int predictedClass;
    private final double[] classProbabilities;
    private final String className;

    public MultipleTrianglePredictionResult(int predictedClass, double confidence,
                                          double[] classProbabilities, String className) {
        super(confidence);
        this.predictedClass = predictedClass;
        this.classProbabilities = classProbabilities;
        this.className = className;
    }

    public int getPredictedClass() {
        return predictedClass;
    }

    public double[] getClassProbabilities() {
        return classProbabilities.clone();
    }

    public String getClassName() {
        return className;
    }

    public boolean isInsideTriangle() {
        return predictedClass > 0;
    }

    public int getTriangleIndex() {
        return isInsideTriangle() ? predictedClass : -1;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("Prediction: ").append(className)
          .append(" (Class ").append(predictedClass).append(")")
          .append("\nConfidence: ").append(String.format("%.4f", getConfidence()))
          .append("\nAll probabilities: ");

        for (int i = 0; i < classProbabilities.length; i++) {
            String classDesc = (i == 0) ? "Outside" : "Triangle " + i;
            sb.append("\n  ").append(classDesc).append(": ")
              .append(String.format("%f", classProbabilities[i]));
        }

        return sb.toString();
    }
}
