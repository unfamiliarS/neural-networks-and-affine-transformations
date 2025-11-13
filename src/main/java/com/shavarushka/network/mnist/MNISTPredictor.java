package com.shavarushka.network.mnist;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import com.shavarushka.network.api.ModelImagePredictor;
import com.shavarushka.network.api.PredictionResult;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

public class MNISTPredictor implements ModelImagePredictor {

    private MultiLayerNetwork model;

    public MNISTPredictor(MultiLayerNetwork model) {
        this.model = model;
    }

    public PredictionResult predict(Object imageData) {
        double[] probabilities = predictProbabilities((double[][]) imageData);
        int predictedDigit = argMax(probabilities);
        double confidence = probabilities[predictedDigit];

        return new MNISTPredictionResult(predictedDigit, confidence, probabilities);
    }

    public PredictionResult predictFromImage(File imageFile) {
        double[][] imageData;
        try {
            imageData = load(imageFile);
            return predict((double[][]) imageData);
        } catch (IOException e) {
            e.printStackTrace();
        }
        return null;
    }

    private double[] predictProbabilities(double[][] imageData) {
        if (imageData.length != 28 || imageData[0].length != 28) {
            throw new IllegalArgumentException("Image must be 28x28 pixels");
        }

        double[] flatArray = flattenAndNormalize(imageData);
        INDArray input = Nd4j.create(flatArray, new int[]{1, 784});
        INDArray output = model.output(input);

        return output.toDoubleVector();
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

    public double[][] flattenImage(double[][] image) {
        int height = image.length;
        int width = image[0].length;
        double[][] flattened = new double[1][height * width];
        
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                flattened[0][i * width + j] = image[i][j];
            }
        }
        
        return flattened;
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
