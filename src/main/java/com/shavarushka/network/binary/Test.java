package com.shavarushka.network.binary;

import java.io.IOException;

public class Test {
    public static void main(String[] args) throws IOException {
        BinaryClassifier classifier = new BinaryClassifier();
        
        int numEpochs = 100;
        int numSamples = 1000;
        
        classifier.train(numEpochs, numSamples);
        classifier.evaluate(100);
        classifier.saveModel("binary_classifier_model.zip");
        
        System.out.println("\n=== Детальные тестовые предсказания ===");
        double[][] testPoints = {
            {1.0, 1.0},    // 1 (1+1=2>0)
            {-1.0, -1.0},  // 0 (-1-1=-2<0)
            {2.0, -1.0},   // 1 (2-1=1>0)
            {-2.0, 1.0},   // 0 (-2+1=-1<0)
            {0.5, -0.4},   // 1 (0.5-0.4=0.1>0)
            {-0.3, 0.2},   // 0 (-0.3+0.2=-0.1<0)
            {-10.0, 0.2}   // 0
        };

        for (double[] point : testPoints) {
            double prediction = classifier.predict(point[0], point[1]);
            double sum = point[0] + point[1];
            String expectedClass = sum >= 0 ? "Класс 1" : "Класс 0";
            String predictedClass = prediction >= 0.5 ? "Класс 1" : "Класс 0";
            System.out.printf("Точка (%.1f, %.1f) сумма=%.1f -> Ожидается: %s, Предсказано: %s (вероятность: %.3f)%n", 
                point[0], point[1], sum, expectedClass, predictedClass, prediction);
        }

        // classifier.loadModel("resources/binary_classifier_model.zip");
        // classifier.printAllWeightsAndBiasesAsArrays();

        // classifier.saveDatasetToCSV(classifier.generateDataSet(numSamples, BinaryClassifier.trainSeed), "dataset.csv");
    }
}
