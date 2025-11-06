package com.shavarushka.network.mnist;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import com.shavarushka.network.api.ModelImagePredictorExtended;
import com.shavarushka.network.api.PredictionResult;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

public class MNISTPredictor implements ModelImagePredictorExtended {

    private MultiLayerNetwork model;

    public MNISTPredictor(MultiLayerNetwork model) {
        this.model = model;
    }

    public double[] predict(double[][] imageData) {
        if (imageData.length != 28 || imageData[0].length != 28) {
            throw new IllegalArgumentException("Image must be 28x28 pixels");
        }

        double[] flatArray = flattenAndNormalize(imageData);
        INDArray input = Nd4j.create(flatArray, new int[]{1, 784});
        INDArray output = model.output(input);

        return output.toDoubleVector();
    }

    public int predictDigit(double[][] imageData) {
        double[] probabilities = predict(imageData);
        return argMax(probabilities);
    }

    public PredictionResult predictDetailed(double[][] imageData) {
        double[] probabilities = predict(imageData);
        int predictedDigit = argMax(probabilities);
        double confidence = probabilities[predictedDigit];

        return new PredictionResult(predictedDigit, confidence, probabilities);
    }

    public int predictFromImage(File imageFile) throws IOException {
        double[][] imageData = load(imageFile);
        return predictDigit(imageData);
    }

    public PredictionResult predictDetailedFromImage(File imageFile) throws IOException {
        double[][] imageData = load(imageFile);
        return predictDetailed(imageData);
    }

    private double[] flattenAndNormalize(double[][] imageData) {
        double[] result = new double[28 * 28];
        int index = 0;

        for (int i = 0; i < 28; i++) {
            for (int j = 0; j < 28; j++) {
                result[index++] = imageData[i][j] / 255.0;
            }
        }
        return result;
    }

    public double[][] load(File imageFile) throws IOException {
        BufferedImage image = ImageIO.read(imageFile);

        double[][] imageData = new double[28][28];

        for (int y = 0; y < 28; y++) {
            for (int x = 0; x < 28; x++) {
                int rgb = image.getRGB(x, y);
                int gray = (rgb >> 16) & 0xFF;
                imageData[y][x] = gray;
            }
        }

        return imageData;
    }

    private int argMax(double[] array) {
        int maxIndex = 0;
        double maxValue = array[0];

        for (int i = 1; i < array.length; i++) {
            if (array[i] > maxValue) {
                maxValue = array[i];
                maxIndex = i;
            }
        }
        return maxIndex;
    }
}
