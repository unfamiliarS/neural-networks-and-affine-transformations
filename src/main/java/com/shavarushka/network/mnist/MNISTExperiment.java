package com.shavarushka.network.mnist;

import com.shavarushka.affine.ScaleAffineTransformation;
import com.shavarushka.network.api.ModelLoader;
import com.shavarushka.network.api.ModelPredictor;
import com.shavarushka.network.api.PredictionResult;
import com.shavarushka.network.api.WeightsManager;
import com.shavarushka.network.api.fabric.MNISTModelFabric;
import com.shavarushka.network.api.fabric.ModelFabric;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;

public class MNISTExperiment {

    private ModelPredictor predictor;

    public MNISTExperiment(ModelPredictor predictor) {
        this.predictor = predictor;
    }

    public void runExperiments(WeightsManager weightsManager, double[][] dataset, int[] trueLabels, double rotationDegrees, String outputPath) {
        try (PrintWriter writer = new PrintWriter(new FileWriter(outputPath))) {
            writer.println("experiment_id,actualImage," +
                         "before_data_and_weigths_rotation_prediction,before_data_and_weigths_rotation_confidence," +
                         "after_data_and_before_weigths_rotation_prediction,after_data_and_before_weigths_rotation_confidence," +
                         "before_data_and_after_weigths_rotation_prediction,before_data_and_after_weigths_rotation_confidence," +
                         "after_data_and_weigths_rotation_prediction,after_data_and_weigths_rotation_confidence," +
                         "after_data_and_before_weigths_rotation_prediction_match,after_data_and_before_weigths_rotation_confidence_change," +
                         "before_data_and_after_weigths_rotation_prediction_match,before_data_and_after_weigths_rotation_confidence_change," +
                         "after_data_and_weigths_rotation_prediction_match,after_data_and_weigths_rotation_confidence_change");

            ScaleAffineTransformation affineTransformation = new ScaleAffineTransformation()
                                                            .scaleFactor(5);
            
            double[][] originalWeights = weightsManager.getLayerWeights(0);
            affineTransformation.setMatrixType(false);
            double[][] rotatedWeights = affineTransformation.transform(originalWeights);
            
            affineTransformation.setMatrixType(true);
            double[][] rotatedDataSet = affineTransformation.transform(dataset);

            for (int i = 0; i < dataset.length; i++) {
                double[] originalData = dataset[i];
                double[] rotatedData = rotatedDataSet[i];
                int imageData = trueLabels[i];
                
                // Исходные точка и веса
                PredictionResult beforeAllRotation = predictor.predict(originalData);
                // Повёрнутая точка и исходные веса
                PredictionResult afterDataBeforeWeigthRotation = predictor.predict(rotatedData);
                
                weightsManager.setLayerWeights(0, rotatedWeights);

                // Повёрнутые точка и веса
                PredictionResult afterAllRotation = predictor.predict(rotatedData);
                // Исходная точка и повёрнутые веса
                PredictionResult beforeDataAfterWeigthRotation = predictor.predict(originalData);

                weightsManager.setLayerWeights(0, originalWeights);
                
                String record = createExperimentRecord(
                    i, imageData, beforeAllRotation, afterAllRotation,
                    afterDataBeforeWeigthRotation, beforeDataAfterWeigthRotation
                );

                writer.println(record);
                
                System.out.println("Выполнено экспериментов: " + i + "/" + dataset.length);
            }
            
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private String createExperimentRecord(int experimentId, int image,
                                        PredictionResult beforeAll, PredictionResult afterAll,
                                        PredictionResult afterDataBeforeWeigth,
                                        PredictionResult beforeDataAfterWeigth) {

        int beforePrediction = beforeAll.getPredictedDigit();
        int afterPrediction = afterAll.getPredictedDigit();
        int afterDataBeforeWeigthPrediction = afterDataBeforeWeigth.getPredictedDigit();
        int beforeDataAfterWeigthPrediction = beforeDataAfterWeigth.getPredictedDigit();

        boolean predictionMatch = beforePrediction == afterPrediction;
        double confidenceChange = afterAll.getConfidence() - beforeAll.getConfidence();
        boolean afterDataBeforeWeigthPredictionMatch = beforePrediction == afterDataBeforeWeigthPrediction;
        double afterDataBeforeWeigthConfidenceChange = afterDataBeforeWeigth.getConfidence() - beforeAll.getConfidence();
        boolean beforeDataAfterWeigthPredictionMatch = beforePrediction == beforeDataAfterWeigthPrediction;
        double beforeDataAfterWeigthConfidenceChange = beforeDataAfterWeigth.getConfidence() - beforeAll.getConfidence();

        return String.format("%d,%d,%d,%f,%d,%f,%d,%f,%d,%f,%b,%f,%b,%f,%b,%f",
            experimentId, image,
            beforePrediction, beforeAll.getConfidence(),
            afterDataBeforeWeigthPrediction, afterDataBeforeWeigth.getConfidence(),
            beforeDataAfterWeigthPrediction, beforeDataAfterWeigth.getConfidence(),
            afterPrediction, afterAll.getConfidence(),
            afterDataBeforeWeigthPredictionMatch, afterDataBeforeWeigthConfidenceChange,
            beforeDataAfterWeigthPredictionMatch, beforeDataAfterWeigthConfidenceChange,
            predictionMatch, confidenceChange
        );
    }

    public static MnistDataSet loadRandomMnistImages(String mnistPath, int imagesPerDigit) {
        double[][] data = new double[imagesPerDigit*10][28*28];
        int[] numbers = new int[imagesPerDigit*10];
        int numbersCntr = 0;

        Random random = new Random(42);

        for (int digit = 0; digit <= 9; digit++) {
            List<String> digitFiles = getImageFilesForDigit(mnistPath, digit);
            
            Collections.shuffle(digitFiles, random);
            int count = Math.min(imagesPerDigit, digitFiles.size());

            for (int i = 0; i < count; i++) {
                data[numbersCntr] = ImageHandler.flattenImage(ImageHandler.load(new File(digitFiles.get(i))));
                numbers[numbersCntr] = digit;
                numbersCntr++;
            }
        }

        return new MnistDataSet(data, numbers);
    }

    private static List<String> getImageFilesForDigit(String mnistPath, int digit) {
        File digitDir = new File(mnistPath);
        File[] files = digitDir.listFiles((dir, name) -> 
            name.startsWith(digit + "_") && name.endsWith(".png")
        );
        
        return Arrays.stream(files)
                .map(File::getAbsolutePath)
                .collect(Collectors.toList());
    }

    public static class MnistDataSet {
        public double[][] flattenImages;
        public int[] numbers;
        
        public MnistDataSet(double[][] flattenImages, int[] numbers) {
            this.flattenImages = flattenImages;
            this.numbers = numbers;
        }
    }

    public static void main(String[] args) {
        ModelFabric fabric = new MNISTModelFabric(ModelLoader.load("src/main/resources/mnist-model.zip"));
        ModelPredictor predictor = fabric.createPredictor();
        WeightsManager weightsManager = fabric.createWeightsManager();
        
        MNISTExperiment experiment = new MNISTExperiment(predictor);

        MnistDataSet data = loadRandomMnistImages("src/main/resources/test", 100);
        
        double[] rotationAngles = {256.52};

        for (double angle : rotationAngles) {
            String outputPath = String.format("mnist_scale_experiment_%.0fdeg.csv", angle);
            experiment.runExperiments(weightsManager, data.flattenImages, data.numbers, angle, outputPath);   
        }
    }
}
