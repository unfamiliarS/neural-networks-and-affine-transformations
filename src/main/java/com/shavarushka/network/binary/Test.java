package com.shavarushka.network.binary;

import java.io.IOException;

public class Test {
    public static void main(String[] args) throws IOException {
        BinaryClassifier classifier = new OneLayerOneClass();

        int numEpochs = 1000;
        int numSamples = 10000;

        classifier.train(numEpochs, numSamples);
        classifier.evaluate(1000);
        classifier.saveModel("binary_1class_classifier_model.zip");

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

        // double[][] testPoints = {
        //     {1.0, 1.0},    // 1 (1+1=2>0)
        //     {-1.0, -1.0},  // 0 (-1-1=-2<0)
        //     {2.0, -1.0},   // 1 (2-1=1>0)
        //     {-2.0, 1.0},   // 0 (-2+1=-1<0)
        //     {0.5, -0.4},   // 1 (0.5-0.4=0.1>0)
        //     {-0.3, 0.2},   // 0 (-0.3+0.2=-0.1<0)
        //     {-0.3, 0.3},   // 2 (-0.3+0.3=0==0)
        // };

        for (double[] point : testPoints) {
            double[] prediction = classifier.predict(point[0], point[1]);
            double sum = point[0] + point[1];
            String expectedClass = sum >= 0 ? "Класс 1" : "Класс 0";
            String predictedClass = prediction[0] >= 0.5 ? "Класс 1" : "Класс 0";
            System.out.printf("Точка (%.1f, %.1f) сумма=%.1f -> Ожидается: %s, Предсказано: %s (вероятность: %.3f)%n",
                point[0], point[1], sum, expectedClass, predictedClass, prediction);
        }

        // for (double[] point : testPoints) {
        //     double[] probabilities = classifier.predict(point[0], point[1]);
        //     double sum = point[0] + point[1];

        //     // Определяем ожидаемый класс
        //     String expectedClass;
        //     if (sum < -0.3)
        //         expectedClass = "Класс 0";
        //     else if (sum > 0.1)
        //         expectedClass = "Класс 1";
        //     else
        //         expectedClass = "Класс 2";

        //     // Определяем предсказанный класс (класс с максимальной вероятностью)
        //     int predictedClassIndex = 0;
        //     double maxProbability = probabilities[0];
        //     for (int i = 1; i < probabilities.length; i++) {
        //         if (probabilities[i] > maxProbability) {
        //             maxProbability = probabilities[i];
        //             predictedClassIndex = i;
        //         }
        //     }

        //     String predictedClass;
        //     switch (predictedClassIndex) {
        //         case 0: predictedClass = "Класс 0"; break;
        //         case 1: predictedClass = "Класс 1"; break;
        //         case 2: predictedClass = "Класс 2"; break;
        //         default: predictedClass = "Неизвестно";
        //     }

        //     System.out.printf("Точка (%.1f, %.1f) сумма=%.1f -> Ожидается: %s, Предсказано: %s%n",
        //             point[0], point[1], sum, expectedClass, predictedClass);
        //     System.out.printf("Вероятности: Класс 0: %.3f, Класс 1: %.3f, Класс 2: %.3f%n",
        //             probabilities[0], probabilities[1], probabilities[2]);
        //     System.out.println("---");
        // }
    }
}
