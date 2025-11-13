package com.shavarushka.network.api;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;

public interface DataGenerator {
    
    DataSet generate();

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
