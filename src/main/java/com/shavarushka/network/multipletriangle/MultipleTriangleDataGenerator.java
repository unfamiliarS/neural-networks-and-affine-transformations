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

    private double[][][] trianglesVertices = {
        {
            {1.0, 1.0},
            {5.0, 1.0},
            {3.0, 4.0}
        },
        {
            {5.0, 5.0},
            {11.0, 5.0},
            {8.0, 11.0}
        }
    };

    public MultipleTriangleDataGenerator(int numSamples, boolean isTrain) {
        this.numSamples = numSamples;
        this.seed = isTrain ? 12345 : 54321;
    }

    @Override
    public DataSet generate() {
        Nd4j.getRandom().setSeed(seed);

        INDArray features = Nd4j.create(numSamples, 2);
        INDArray labels = Nd4j.create(numSamples, 1);

        int samplesPerClass = numSamples / 2;
        int insideCount = 0;
        int outsideCount = 0;

        while (insideCount < samplesPerClass || outsideCount < samplesPerClass) {
            double x = Nd4j.getRandom().nextDouble() * 12.0;
            double y = Nd4j.getRandom().nextDouble() * 12.0;

            boolean inAnyTriangle = isPointInAnyTriangle(x, y);

            if (inAnyTriangle && insideCount < samplesPerClass) {
                int index = insideCount + outsideCount;
                features.putScalar(new int[]{index, 0}, x);
                features.putScalar(new int[]{index, 1}, y);
                labels.putScalar(new int[]{index, 0}, 1.0);
                insideCount++;
            } else if (!inAnyTriangle && outsideCount < samplesPerClass) {
                int index = insideCount + outsideCount;
                features.putScalar(new int[]{index, 0}, x);
                features.putScalar(new int[]{index, 1}, y);
                labels.putScalar(new int[]{index, 0}, 0.0);
                outsideCount++;
            }
        }

        return shuffleDataset(new DataSet(features, labels));
    }

    private boolean isPointInAnyTriangle(double x, double y) {
        return isPointInTriangle(x, y, trianglesVertices[0]) ||
               isPointInTriangle(x, y, trianglesVertices[1]);
    }

    private boolean isPointInTriangle(double x, double y, double[][] triangle) {
        double x1 = triangle[0][0], y1 = triangle[0][1];
        double x2 = triangle[1][0], y2 = triangle[1][1];
        double x3 = triangle[2][0], y3 = triangle[2][1];

        double denominator = ((y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3));
        double a = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / denominator;
        double b = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / denominator;
        double c = 1 - a - b;

        return a >= 0 && a <= 1 && b >= 0 && b <= 1 && c >= 0 && c <= 1;
    }

    private DataSet shuffleDataset(DataSet dataset) {
        int numSamples = (int) dataset.getFeatures().size(0);
        int[] indices = new int[numSamples];
        for (int i = 0; i < numSamples; i++)
            indices[i] = i;

        Random random = new Random(seed);
        for (int i = numSamples - 1; i > 0; i--) {
            int j = random.nextInt(i + 1);
            int temp = indices[i];
            indices[i] = indices[j];
            indices[j] = temp;
        }

        INDArray shuffledFeatures = Nd4j.create(numSamples, 2);
        INDArray shuffledLabels = Nd4j.create(numSamples, 1);

        for (int i = 0; i < numSamples; i++) {
            int originalIndex = indices[i];
            shuffledFeatures.putRow(i, dataset.getFeatures().getRow(originalIndex));
            shuffledLabels.putRow(i, dataset.getLabels().getRow(originalIndex));
        }

        return new DataSet(shuffledFeatures, shuffledLabels);
    }

    public static void saveToCSV(DataSet dataset, String filename) {
        try (PrintWriter writer = new PrintWriter(new FileWriter(filename))) {
            writer.println("x,y,class,class_name");

            INDArray features = dataset.getFeatures();
            INDArray labels = dataset.getLabels();

            for (int i = 0; i < features.rows(); i++) {
                double x = features.getDouble(i, 0);
                double y = features.getDouble(i, 1);
                double label = labels.getDouble(i, 0);
                int classIndex = label > 0.5 ? 1 : 0;
                String className = classIndex == 1 ? "inside" : "outside";

                writer.printf("%.6f,%.6f,%d,%s%n", x, y, classIndex, className);
            }

            System.out.println("Датасет треугольников сохранен в: " + filename);
        } catch (IOException e) {
            e.printStackTrace();
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
                    double x = Double.parseDouble(parts[0].trim());
                    double y = Double.parseDouble(parts[1].trim());

                    pointsList.add(new double[]{x, y});
                }
            }

        } catch (IOException | NumberFormatException e) {
            e.printStackTrace();
        }

        double[][] pointsArray = new double[pointsList.size()][2];
        for (int i = 0; i < pointsList.size(); i++) {
            pointsArray[i] = pointsList.get(i);
        }

        return pointsArray;
    }
}
