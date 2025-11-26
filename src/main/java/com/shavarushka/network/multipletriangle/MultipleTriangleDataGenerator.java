package com.shavarushka.network.multipletriangle;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import com.shavarushka.network.api.DataGenerator;

public class MultipleTriangleDataGenerator implements DataGenerator {

    private long seed;
    private int numSamples;
    private int numTriangles;
    private double[][][] trianglesVertices;

    public MultipleTriangleDataGenerator(int numSamples, int numTriangles, boolean isTrain) {
        this.numSamples = numSamples;
        this.numTriangles = numTriangles;
        this.seed = isTrain ? 12345 : 54321;
        this.trianglesVertices = generateTriangles(numTriangles);
    }

    @Override
    public DataSet generate() {
        Nd4j.getRandom().setSeed(seed);

        INDArray features = Nd4j.create(numSamples, 2);
        INDArray labels = Nd4j.create(numSamples, numTriangles + 1);

        int totalClasses = numTriangles + 1;
        int samplesPerClass = numSamples / totalClasses;
        int[] counts = new int[totalClasses];

        // Убедимся, что общее количество samples не превышает numSamples
        int actualTotalSamples = samplesPerClass * totalClasses;
        if (actualTotalSamples > numSamples) {
            samplesPerClass = numSamples / totalClasses;
            actualTotalSamples = samplesPerClass * totalClasses;
        }

        Random random = new Random(seed);
        int totalCount = 0;

        while (totalCount < actualTotalSamples) {
            double x = random.nextDouble() * 10.0;
            double y = random.nextDouble() * 10.0;

            int triangleIndex = -1;
            for (int i = 0; i < numTriangles; i++) {
                if (isPointInTriangle(x, y, trianglesVertices[i])) {
                    triangleIndex = i;
                    break;
                }
            }

            int classIndex = triangleIndex + 1;

            // Проверяем, нужны ли еще samples для этого класса
            if (classIndex >= 0 && classIndex < counts.length && counts[classIndex] < samplesPerClass) {
                addSample(features, labels, totalCount, x, y, classIndex, totalClasses);
                counts[classIndex]++;
                totalCount++;
            }

            // Защита от бесконечного цикла
            if (totalCount > numSamples) {
                break;
            }
        }

        // Если не набрали достаточно samples, заполним оставшиеся случайными точками
        while (totalCount < numSamples) {
            double x = random.nextDouble() * 10.0;
            double y = random.nextDouble() * 10.0;

            // Определяем класс для случайной точки
            int triangleIndex = -1;
            for (int i = 0; i < numTriangles; i++) {
                if (isPointInTriangle(x, y, trianglesVertices[i])) {
                    triangleIndex = i;
                    break;
                }
            }
            int classIndex = triangleIndex + 1;

            addSample(features, labels, totalCount, x, y, classIndex, totalClasses);
            totalCount++;
        }

        return shuffleDataset(features, labels);
    }

    private double[][][] generateTriangles(int numTriangles) {
        double[][][] triangles = new double[numTriangles][3][2];
        Random random = new Random(seed);

        for (int i = 0; i < numTriangles; i++) {
            double centerX = 2.0 + random.nextDouble() * 6.0;
            double centerY = 2.0 + random.nextDouble() * 6.0;
            double size = 0.5 + random.nextDouble() * 2.0;

            triangles[i][0] = new double[]{centerX, centerY + size};
            triangles[i][1] = new double[]{centerX - size, centerY - size};
            triangles[i][2] = new double[]{centerX + size, centerY - size};
        }

        return triangles;
    }

    private boolean isPointInTriangle(double x, double y, double[][] triangle) {
        double x1 = triangle[0][0], y1 = triangle[0][1];
        double x2 = triangle[1][0], y2 = triangle[1][1];
        double x3 = triangle[2][0], y3 = triangle[2][1];

        double denominator = ((y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3));
        double a = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / denominator;
        double b = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / denominator;
        double c = 1 - a - b;

        return a >= 0 && b >= 0 && c >= 0;
    }

    private void addSample(INDArray features, INDArray labels, int index, double x, double y, int classIndex, int numClasses) {
        // Проверяем, что индекс в пределах границ
        if (index >= features.rows()) {
            throw new IllegalArgumentException("Index " + index + " is out of bounds for array with " + features.rows() + " rows");
        }

        features.putScalar(new int[]{index, 0}, x);
        features.putScalar(new int[]{index, 1}, y);

        // One-hot encoding
        for (int i = 0; i < numClasses; i++) {
            labels.putScalar(new int[]{index, i}, i == classIndex ? 1.0 : 0.0);
        }
    }

    private DataSet shuffleDataset(INDArray features, INDArray labels) {
        int numSamples = features.rows();
        int[] indices = new int[numSamples];
        for (int i = 0; i < numSamples; i++) {
            indices[i] = i;
        }

        Random random = new Random(seed);
        for (int i = numSamples - 1; i > 0; i--) {
            int j = random.nextInt(i + 1);
            int temp = indices[i];
            indices[i] = indices[j];
            indices[j] = temp;
        }

        INDArray shuffledFeatures = Nd4j.create(numSamples, 2);
        INDArray shuffledLabels = Nd4j.create(numSamples, labels.columns());

        for (int i = 0; i < numSamples; i++) {
            int originalIndex = indices[i];
            shuffledFeatures.putRow(i, features.getRow(originalIndex));
            shuffledLabels.putRow(i, labels.getRow(originalIndex));
        }

        return new DataSet(shuffledFeatures, shuffledLabels);
    }

    public double[][][] getTrianglesVertices() {
        return trianglesVertices;
    }

    public static void saveToCSV(DataSet dataset, String filename, int numTriangles) {
        try (PrintWriter writer = new PrintWriter(new FileWriter(filename))) {
            writer.println("x,y,class,class_name");

            INDArray features = dataset.getFeatures();
            INDArray labels = dataset.getLabels();

            for (int i = 0; i < features.rows(); i++) {
                double x = features.getDouble(i, 0);
                double y = features.getDouble(i, 1);

                int classIndex = -1;
                double maxProb = -1.0;
                for (int j = 0; j < labels.columns(); j++) {
                    double prob = labels.getDouble(i, j);
                    if (prob > maxProb) {
                        maxProb = prob;
                        classIndex = j;
                    }
                }

                String className = getClassName(classIndex, numTriangles);

                writer.printf("%.6f,%.6f,%d,%s%n", x, y, classIndex, className);
            }

            System.out.println("Датасет треугольников сохранен в: " + filename);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static String getClassName(int classIndex, int numTriangles) {
        if (classIndex == 0) {
            return "outside_all";
        } else if (classIndex <= numTriangles) {
            return "triangle_" + classIndex;
        } else {
            return "unknown";
        }
    }

    public static double[][] getFromCSV(String filename) {
        List<double[]> pointsList = new ArrayList<>();

        try (BufferedReader reader = new BufferedReader(new FileReader(filename))) {
            String line;
            boolean isFirstLine = true;

            while ((line = reader.readLine()) != null) {
                if (isFirstLine) {
                    isFirstLine = false;
                    continue;
                }

                String[] parts = line.split(",");
                if (parts.length >= 2) {
                    // Формат: x,y,class,class_name - берем только x и y
                    double x = Double.parseDouble(parts[0].trim());
                    double y = Double.parseDouble(parts[1].trim());

                    pointsList.add(new double[]{x, y});
                }
            }

        } catch (IOException | NumberFormatException e) {
            e.printStackTrace();
        }

        // Конвертируем List в double[][]
        double[][] pointsArray = new double[pointsList.size()][2];
        for (int i = 0; i < pointsList.size(); i++) {
            pointsArray[i] = pointsList.get(i);
        }

        return pointsArray;
    }
}
