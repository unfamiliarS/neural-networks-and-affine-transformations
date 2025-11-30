package com.shavarushka.network.multipletriangle;

import com.shavarushka.affine.AffineTransformation;
import com.shavarushka.affine.RotationAffineTransformation;
import com.shavarushka.network.api.ModelLoader;
import com.shavarushka.network.api.ModelPredictor;
import com.shavarushka.network.api.PredictionResult;
import com.shavarushka.network.api.WeightsManager;
import com.shavarushka.network.api.fabric.ModelFabric;
import com.shavarushka.network.api.fabric.MultipleTriangleModelFabric;

import java.io.BufferedReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;

public class TriangleExperiment {

    private ModelPredictor predictor;

    public TriangleExperiment(ModelPredictor predictor) {
        this.predictor = predictor;
    }

    public void runExperiments(WeightsManager weightsManager, double[][] dataset, byte[] trueLabels, double rotationDegrees, String outputPath) {
        try (PrintWriter writer = new PrintWriter(new FileWriter(outputPath))) {
            writer.println("experiment_id,x,y,true_label," +
                         "before_data_and_weigths_transformation_prediction,before_data_and_weigths_transformation_confidence," +
                         "after_data_and_before_weigths_transformation_prediction,after_data_and_before_weigths_transformation_confidence," +
                         "before_data_and_after_weigths_transformation_prediction,before_data_and_after_weigths_transformation_confidence," +
                         "after_data_and_weigths_transformation_prediction,after_data_and_weigths_transformation_confidence," +
                         "after_data_and_before_weigths_transformation_prediction_match,after_data_and_before_weigths_transformation_confidence_change," +
                         "before_data_and_after_weigths_transformation_prediction_match,before_data_and_after_weigths_transformation_confidence_change," +
                         "after_data_and_weigths_transformation_prediction_match,after_data_and_weigths_transformation_confidence_change");

            AffineTransformation affineTransformation = new RotationAffineTransformation().setAngle(256);

            double[][] originalWeights = weightsManager.getLayerWeights(0);
            double[][] rotatedWeights = affineTransformation.transform(originalWeights);
            
            double[][] rotatedDataSet = affineTransformation.transform(dataset);

            for (int i = 0; i < dataset.length; i++) {
                double[] originalPoint = dataset[i];
                double[] rotatedPoint = rotatedDataSet[i];
                byte trueLabel = trueLabels[i];
                
                // Исходные точка и веса
                PredictionResult beforeAllTransformation = predictor.predict(originalPoint);
                // Повёрнутая точка и исходные веса
                PredictionResult afterDataBeforeWeigthTransformation = predictor.predict(rotatedPoint);
                
                weightsManager.setLayerWeights(0, rotatedWeights);

                // Повёрнутые точка и веса
                PredictionResult afterAllTransformation = predictor.predict(rotatedPoint);
                // Исходная точка и повёрнутые веса
                PredictionResult beforeDataAfterWeigthTransformation = predictor.predict(originalPoint);

                weightsManager.setLayerWeights(0, originalWeights);
                
                String record = createExperimentRecord(
                    i, originalPoint, trueLabel, beforeAllTransformation, afterAllTransformation,
                    afterDataBeforeWeigthTransformation, beforeDataAfterWeigthTransformation
                );

                writer.println(record);
                
                if (i % 100 == 0) {
                    System.out.println("Выполнено экспериментов: " + i + "/" + dataset.length);
                }
            }
            
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private String createExperimentRecord(int experimentId, double[] point, byte trueLabel,
                                        PredictionResult beforeAll, PredictionResult afterAll,
                                        PredictionResult afterDataBeforeWeigth,
                                        PredictionResult beforeDataAfterWeigth) {
        double x = point[0];
        double y = point[1];

        String beforePrediction = beforeAll.getConfidence() > 0.5 ? "in" : "out";
        String afterPrediction = afterAll.getConfidence() > 0.5 ? "in" : "out";
        String afterDataBeforeWeigthPrediction = afterDataBeforeWeigth.getConfidence() > 0.5 ? "in" : "out";
        String beforeDataAfterWeigthPrediction = beforeDataAfterWeigth.getConfidence() > 0.5 ? "in" : "out";

        boolean predictionMatch = beforePrediction.equals(afterPrediction);
        double confidenceChange = afterAll.getConfidence() - beforeAll.getConfidence();
        boolean afterDataBeforeWeigthPredictionMatch = beforePrediction.equals(afterDataBeforeWeigthPrediction);
        double afterDataBeforeWeigthConfidenceChange = afterDataBeforeWeigth.getConfidence() - beforeAll.getConfidence();
        boolean beforeDataAfterWeigthPredictionMatch = beforePrediction.equals(beforeDataAfterWeigthPrediction);
        double beforeDataAfterWeigthConfidenceChange = beforeDataAfterWeigth.getConfidence() - beforeAll.getConfidence();

        return String.format("%d,%f,%f,%d,%s,%f,%s,%f,%s,%f,%s,%f,%b,%f,%b,%f,%b,%f",
            experimentId, x, y, trueLabel,
            beforePrediction, beforeAll.getConfidence(),
            afterDataBeforeWeigthPrediction, afterDataBeforeWeigth.getConfidence(),
            beforeDataAfterWeigthPrediction, beforeDataAfterWeigth.getConfidence(),
            afterPrediction, afterAll.getConfidence(),
            afterDataBeforeWeigthPredictionMatch, afterDataBeforeWeigthConfidenceChange,
            beforeDataAfterWeigthPredictionMatch, beforeDataAfterWeigthConfidenceChange,
            predictionMatch, confidenceChange
        );
    }

    public static DatasetWithLabels loadDatasetWithLabels(String csvPath) {
        List<double[]> points = new ArrayList<>();
        List<Byte> labels = new ArrayList<>();

        try (BufferedReader reader = new BufferedReader(new FileReader(csvPath))) {
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
                    byte label = Byte.parseByte(parts[2].trim());

                    points.add(new double[]{x, y});
                    labels.add(label);
                }
            }
        } catch (NumberFormatException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
        
        double[][] pointsArray = points.toArray(new double[0][]);
        byte[] labelsArray = new byte[labels.size()];
        for (int i = 0; i < labels.size(); i++)
            labelsArray[i] = labels.get(i);
        
        return new DatasetWithLabels(pointsArray, labelsArray);
    }

    public static class DatasetWithLabels {
        public double[][] points;
        public byte[] labels;
        
        public DatasetWithLabels(double[][] points, byte[] labels) {
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
        
        double[] rotationAngles = {256.52};

        for (double angle : rotationAngles) {
            String outputPath = String.format("two_triangle_experiment_rotate_%.0fdeg.csv", angle);
            experiment.runExperiments(weightsManager, data.points, data.labels, angle, outputPath);   
        }
    }
}
