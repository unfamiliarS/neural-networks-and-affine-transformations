package com.shavarushka.network.triangle;

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

public class TriangleDataGenerator implements DataGenerator {

    private long seed;
    private int numSamples;

    private double[][] triangleVertices = {
        {1.0, 1.0},
        {5.0, 1.0},
        {3.0, 4.0}
    };

    public TriangleDataGenerator(int numSamples, boolean isTrain) {
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
            double x = Nd4j.getRandom().nextDouble() * 6.0;
            double y = Nd4j.getRandom().nextDouble() * 6.0;

            boolean insideTriangle = isPointInTriangle(x, y);

            if (insideTriangle && insideCount < samplesPerClass) {
                int index = insideCount + outsideCount;
                features.putScalar(new int[]{index, 0}, x);
                features.putScalar(new int[]{index, 1}, y);
                labels.putScalar(new int[]{index, 0}, 1.0);
                insideCount++;
            } else if (!insideTriangle && outsideCount < samplesPerClass) {
                int index = insideCount + outsideCount;
                features.putScalar(new int[]{index, 0}, x);
                features.putScalar(new int[]{index, 1}, y);
                labels.putScalar(new int[]{index, 0}, 0.0);
                outsideCount++;
            }
        }

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
            shuffledFeatures.putRow(i, features.getRow(originalIndex));
            shuffledLabels.putRow(i, labels.getRow(originalIndex));
        }

        return new DataSet(shuffledFeatures, shuffledLabels);
    }

    private boolean isPointInTriangle(double x, double y) {
        double x1 = triangleVertices[0][0], y1 = triangleVertices[0][1];
        double x2 = triangleVertices[1][0], y2 = triangleVertices[1][1];
        double x3 = triangleVertices[2][0], y3 = triangleVertices[2][1];

        double denominator = ((y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3));
        double a = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / denominator;
        double b = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / denominator;
        double c = 1 - a - b;

        return a >= 0 && b >= 0 && c >= 0;
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
                int classIndex = label > 0.5 ? 0 : 1;
                String className = classIndex == 0 ? "inside" : "outside";

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
