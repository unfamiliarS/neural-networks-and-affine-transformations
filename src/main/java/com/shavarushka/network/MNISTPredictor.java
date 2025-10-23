package com.shavarushka.network;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

public class MNISTPredictor {
    private final MultiLayerNetwork model;

    public MNISTPredictor(MultiLayerNetwork model) {
        this.model = model;
    }

    /**
     * Классификация изображения MNIST
     * @param imageData массив 28x28 с значениями пикселей (0-255)
     * @return массив вероятностей для каждой цифры (0-9)
     */
    public double[] predict(double[][] imageData) {
        if (imageData.length != 28 || imageData[0].length != 28) {
            throw new IllegalArgumentException("Image must be 28x28 pixels");
        }

        // Преобразуем в 1D массив и нормализуем
        double[] flatArray = flattenAndNormalize(imageData);
        
        // Создаем INDArray
        INDArray input = Nd4j.create(flatArray, new int[]{1, 784});
        
        // Получаем предсказание
        INDArray output = model.output(input);
        
        // Преобразуем в массив double
        return output.toDoubleVector();
    }

    /**
     * Классификация с возвратом наиболее вероятной цифры
     * @param imageData массив 28x28 с значениями пикселей (0-255)
     * @return предсказанная цифра (0-9)
     */
    public int predictDigit(double[][] imageData) {
        double[] probabilities = predict(imageData);
        return argMax(probabilities);
    }

    /**
     * Классификация с детальной информацией
     * @param imageData массив 28x28 с значениями пикселей (0-255)
     * @return объект PredictionResult с деталями предсказания
     */
    public PredictionResult predictDetailed(double[][] imageData) {
        double[] probabilities = predict(imageData);
        int predictedDigit = argMax(probabilities);
        double confidence = probabilities[predictedDigit];
        
        return new PredictionResult(predictedDigit, confidence, probabilities);
    }

    /**
     * Загрузка и классификация изображения из файла
     * @param imageFile файл с изображением
     * @return предсказанная цифра
     */
    public int predictFromImage(File imageFile) throws IOException {
        double[][] imageData = loadAndPreprocessImage(imageFile);
        return predictDigit(imageData);
    }

    /**
     * Загрузка и классификация с детальной информацией
     * @param imageFile файл с изображением
     * @return объект PredictionResult с деталями предсказания
     */
    public PredictionResult predictDetailedFromImage(File imageFile) throws IOException {
        double[][] imageData = loadAndPreprocessImage(imageFile);
        return predictDetailed(imageData);
    }

    /**
     * Преобразование 2D массива в 1D с нормализацией
     */
    private double[] flattenAndNormalize(double[][] imageData) {
        double[] result = new double[28 * 28];
        int index = 0;
        
        for (int i = 0; i < 28; i++) {
            for (int j = 0; j < 28; j++) {
                // Нормализуем в диапазон [0, 1]
                result[index++] = imageData[i][j] / 255.0;
            }
        }
        return result;
    }

    /**
     * Загрузка и предобработка изображения
     */
    private double[][] loadAndPreprocessImage(File imageFile) throws IOException {
        BufferedImage image = ImageIO.read(imageFile);
        
        // Конвертируем в grayscale и изменяем размер до 28x28 если нужно
        BufferedImage processedImage = preprocessImage(image);
        
        double[][] imageData = new double[28][28];
        
        for (int y = 0; y < 28; y++) {
            for (int x = 0; x < 28; x++) {
                int rgb = processedImage.getRGB(x, y);
                // Извлекаем значение яркости (предполагаем grayscale)
                int gray = (rgb >> 16) & 0xFF; // Берем красный канал для grayscale
                imageData[y][x] = gray;
            }
        }
        
        return imageData;
    }

    /**
     * Предобработка изображения: конвертация в grayscale и resize до 28x28
     */
    private BufferedImage preprocessImage(BufferedImage image) {
        // Если изображение уже 28x28 и в grayscale, возвращаем как есть
        if (image.getWidth() == 28 && image.getHeight() == 28 && 
            image.getType() == BufferedImage.TYPE_BYTE_GRAY) {
            return image;
        }
        
        // Создаем новое изображение 28x28 grayscale
        BufferedImage processed = new BufferedImage(28, 28, BufferedImage.TYPE_BYTE_GRAY);
        java.awt.Graphics2D g = processed.createGraphics();
        
        // Масштабируем исходное изображение до 28x28
        g.drawImage(image, 0, 0, 28, 28, null);
        g.dispose();
        
        return processed;
    }

    /**
     * Нахождение индекса максимального значения в массиве
     */
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

    /**
     * Вспомогательный метод для создания тестового изображения (для демонстрации)
     */
    public static double[][] createTestImage(int digit) {
        double[][] image = new double[28][28];
        // Заполняем случайными значениями (в реальности здесь была бы логика создания цифры)
        for (int i = 0; i < 28; i++) {
            for (int j = 0; j < 28; j++) {
                image[i][j] = Math.random() * 255;
            }
        }
        return image;
    }

    /**
     * Класс для хранения результатов предсказания
     */
    public static class PredictionResult {
        private final int predictedDigit;
        private final double confidence;
        private final double[] probabilities;

        public PredictionResult(int predictedDigit, double confidence, double[] probabilities) {
            this.predictedDigit = predictedDigit;
            this.confidence = confidence;
            this.probabilities = probabilities.clone();
        }

        public int getPredictedDigit() {
            return predictedDigit;
        }

        public double getConfidence() {
            return confidence;
        }

        public double[] getProbabilities() {
            return probabilities.clone();
        }

        public void printDetails() {
            System.out.println("Predicted digit: " + predictedDigit);
            System.out.printf("Confidence: %.4f%n", confidence);
            System.out.println("All probabilities:");
            for (int i = 0; i < probabilities.length; i++) {
                System.out.printf("  %d: %.4f%n", i, probabilities[i]);
            }
        }
    }
}
