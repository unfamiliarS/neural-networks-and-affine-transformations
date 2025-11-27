package com.shavarushka.network.multipletriangle;

import com.shavarushka.affine.AffineTransformations;
import com.shavarushka.network.api.ModelLoader;
import com.shavarushka.network.api.ModelPredictor;
import com.shavarushka.network.api.PredictionResult;
import com.shavarushka.network.api.WeightsManager;
import com.shavarushka.network.api.fabric.ModelFabric;
import com.shavarushka.network.api.fabric.MultipleTriangleModelFabric;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;

public class TriangleExperiment {

    private ModelPredictor predictor;
    private SimpleDateFormat dateFormat;

    public TriangleExperiment(ModelPredictor predictor) {
        this.predictor = predictor;
        this.dateFormat = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
    }

    public void runExperiments(WeightsManager weightsManager, double[][] dataset, double[] trueLabels, double rotationDegrees, String outputPath) {
        try (PrintWriter writer = new PrintWriter(new FileWriter(outputPath))) {
            writer.println("experiment_id,timestamp,x,y,true_label," +
                         "before_rotation_prediction,before_rotation_confidence," +
                         "after_rotation_prediction,after_rotation_confidence," +
                         "prediction_match,confidence_change");

            double[][] originalWeights = weightsManager.getLayerWeights(0);
            double[][] rotatedWeights = AffineTransformations.rotateComplex(originalWeights, rotationDegrees);
            
            double[][] rotatedDataSet = AffineTransformations.rotateComplex(dataset, rotationDegrees);

            for (int i = 0; i < dataset.length; i++) {
                double[] originalPoint = dataset[i];
                double[] rotatedPoint = rotatedDataSet[i];
                double trueLabel = trueLabels[i];
                
                // Предсказание для исходной точки
                PredictionResult beforeRotation = predictor.predict(originalPoint);
                
                weightsManager.setLayerWeights(0, rotatedWeights);

                // Предсказание для повернутой точки
                PredictionResult afterRotation = predictor.predict(rotatedPoint);

                weightsManager.setLayerWeights(0, originalWeights);
                
                String record = createExperimentRecord(
                    i, originalPoint, trueLabel, beforeRotation, afterRotation
                );
                writer.println(record);
                
                if (i % 100 == 0) {
                    System.out.println("Выполнено экспериментов: " + i + "/" + dataset.length);
                }
            }
            
            System.out.println("Все эксперименты завершены. Результаты сохранены в: " + outputPath);
            
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private String createExperimentRecord(int experimentId, double[] point, double trueLabel,
                                        PredictionResult before, PredictionResult after) {
        double x = point[0];
        double y = point[1];
        String timestamp = dateFormat.format(new Date());

        String beforePrediction = before.getConfidence() > 0.5 ? "in" : "out";
        String afterPrediction = after.getConfidence() > 0.5 ? "in" : "out";

        boolean predictionMatch = beforePrediction.equals(afterPrediction);
        double confidenceChange = after.getConfidence() - before.getConfidence();

        return String.format("%d,%s,%.6f,%.6f,%.1f,%s,%s,%s,%s,%b,%.6f",
            experimentId, timestamp, x, y, trueLabel,
            beforePrediction, before.getConfidence(),
            afterPrediction, after.getConfidence(),
            predictionMatch, confidenceChange
        );
    }

    public static DatasetWithLabels loadDatasetWithLabels(String csvPath) {
        try {
            java.util.List<double[]> points = new java.util.ArrayList<>();
            java.util.List<Double> labels = new java.util.ArrayList<>();
            java.io.BufferedReader reader = new java.io.BufferedReader(
                new java.io.FileReader(csvPath));

            String line;
            boolean firstLine = true;
            while ((line = reader.readLine()) != null) {
                if (firstLine) {
                    firstLine = false;
                    continue;
                }
                String[] parts = line.split(",");
                if (parts.length >= 4) {
                    double x = Double.parseDouble(parts[0].trim());
                    double y = Double.parseDouble(parts[1].trim());
                    double label = Double.parseDouble(parts[2].trim());
                    
                    points.add(new double[]{x, y});
                    labels.add(label);
                }
            }
            reader.close();
            
            double[][] pointsArray = points.toArray(new double[0][]);
            double[] labelsArray = new double[labels.size()];
            for (int i = 0; i < labels.size(); i++) {
                labelsArray[i] = labels.get(i);
            }
            
            return new DatasetWithLabels(pointsArray, labelsArray);
            
        } catch (IOException e) {
            e.printStackTrace();
            return new DatasetWithLabels(new double[0][], new double[0]);
        }
    }

    public static class DatasetWithLabels {
        public double[][] points;
        public double[] labels;
        
        public DatasetWithLabels(double[][] points, double[] labels) {
            this.points = points;
            this.labels = labels;
        }
    }

    public static void main(String[] args) {
        ModelFabric fabric = new MultipleTriangleModelFabric(ModelLoader.load("src/main/resources/two-triangles.zip"));
        ModelPredictor predictor = fabric.createPredictor();
        WeightsManager weightsManager = fabric.createWeightsManager();
        
        TriangleExperiment experiment = new TriangleExperiment(predictor);
        
        DatasetWithLabels data = loadDatasetWithLabels("src/main/python/multipletriangle/dataset.csv");
        
        double[] rotationAngles = {30.0, 45.0, 124.0, 256.0};
        
        for (double angle : rotationAngles) {
            String outputPath = String.format("experiment_results_%.0fdeg.csv", angle);
            System.out.println("\nЗапуск экспериментов с поворотом на " + angle + " градусов...");
            experiment.runExperiments(weightsManager, data.points, data.labels, angle, outputPath);
            
            ExperimentAnalyzer.analyzeResults(outputPath, angle);
        }
    }
}

class ExperimentAnalyzer {
    
    public static void analyzeResults(String resultsPath, double rotationAngle) {
        try (BufferedReader reader = new BufferedReader(new FileReader(resultsPath))) {
            String line;
            List<ExperimentRecord> records = new ArrayList<>();
            
            reader.readLine();
            
            while ((line = reader.readLine()) != null) {
                String[] parts = line.split(",");
                if (parts.length >= 11) {
                    ExperimentRecord record = new ExperimentRecord(parts);
                    records.add(record);
                }
            }
            
            printAnalysis(records, rotationAngle);
            
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
    
    private static void printAnalysis(List<ExperimentRecord> records, double rotationAngle) {
        int total = records.size();
        int predictionMatches = 0;
        int correctBefore = 0;
        int correctAfter = 0;
        double totalConfidenceChange = 0;
        double avgConfidenceBefore = 0;
        double avgConfidenceAfter = 0;
        
        for (ExperimentRecord record : records) {
            if (record.predictionMatch) predictionMatches++;
            
            boolean isCorrectBefore = record.beforePrediction.equals(
                record.trueLabel > 0.5 ? "in" : "out");
            boolean isCorrectAfter = record.afterPrediction.equals(
                record.trueLabel > 0.5 ? "in" : "out");
            
            if (isCorrectBefore) correctBefore++;
            if (isCorrectAfter) correctAfter++;
            
            totalConfidenceChange += record.confidenceChange;
            avgConfidenceBefore += record.beforeConfidence;
            avgConfidenceAfter += record.afterConfidence;
        }
        
        avgConfidenceBefore /= total;
        avgConfidenceAfter /= total;
        
        System.out.println("\n=== АНАЛИЗ РЕЗУЛЬТАТОВ (поворот " + rotationAngle + "°) ===");
        System.out.println("Всего экспериментов: " + total);
        System.out.printf("Точность до поворота: %.2f%%\n", (correctBefore * 100.0 / total));
        System.out.printf("Точность после поворота: %.2f%%\n", (correctAfter * 100.0 / total));
        System.out.printf("Совпадение предсказаний: %.2f%%\n", (predictionMatches * 100.0 / total));
        System.out.printf("Средняя уверенность до: %.6f\n", avgConfidenceBefore);
        System.out.printf("Средняя уверенность после: %.6f\n", avgConfidenceAfter);
        System.out.printf("Среднее изменение уверенности: %.6f\n", (totalConfidenceChange / total));
        
        analyzeByClass(records);
    }
    
    private static void analyzeByClass(List<ExperimentRecord> records) {
        int insidePoints = 0;
        int outsidePoints = 0;
        int insideMatches = 0;
        int outsideMatches = 0;
        double insideConfidenceChange = 0;
        double outsideConfidenceChange = 0;
        
        for (ExperimentRecord record : records) {
            if (record.trueLabel > 0.5) {
                insidePoints++;
                if (record.predictionMatch) insideMatches++;
                insideConfidenceChange += record.confidenceChange;
            } else {
                outsidePoints++;
                if (record.predictionMatch) outsideMatches++;
                outsideConfidenceChange += record.confidenceChange;
            }
        }
        
        System.out.println("\n--- Анализ по классам ---");
        System.out.printf("Точки внутри треугольников: %d\n", insidePoints);
        System.out.printf("Совпадение предсказаний для 'inside': %.2f%%\n", 
                         (insideMatches * 100.0 / insidePoints));
        System.out.printf("Среднее изменение уверенности для 'inside': %.6f\n",
                         (insideConfidenceChange / insidePoints));
        
        System.out.printf("Точки снаружи треугольников: %d\n", outsidePoints);
        System.out.printf("Совпадение предсказаний для 'outside': %.2f%%\n", 
                         (outsideMatches * 100.0 / outsidePoints));
        System.out.printf("Среднее изменение уверенности для 'outside': %.6f\n",
                         (outsideConfidenceChange / outsidePoints));
    }
    
    private static class ExperimentRecord {
        int experimentId;
        double x, y;
        double trueLabel;
        String beforePrediction, afterPrediction;
        double beforeConfidence, afterConfidence;
        boolean predictionMatch;
        double confidenceChange;
        
        ExperimentRecord(String[] parts) {
            this.experimentId = Integer.parseInt(parts[0]);
            this.x = Double.parseDouble(parts[2]);
            this.y = Double.parseDouble(parts[3]);
            this.trueLabel = Double.parseDouble(parts[4]);
            this.beforePrediction = parts[5];
            this.beforeConfidence = Double.parseDouble(parts[6]);
            this.afterPrediction = parts[7];
            this.afterConfidence = Double.parseDouble(parts[8]);
            this.predictionMatch = Boolean.parseBoolean(parts[9]);
            this.confidenceChange = Double.parseDouble(parts[10]);
        }
    }
}
