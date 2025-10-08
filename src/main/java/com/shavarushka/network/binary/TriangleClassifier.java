package com.shavarushka.network.binary;

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.Random;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class TriangleClassifier {

    protected MultiLayerNetwork model;
    protected static final int modelSeed = 67890;
    public static final int trainSeed = 12345;
    public static final int validationSeed = 12345;
    
    private double[][] triangleVertices = {
        {1.0, 1.0},
        {5.0, 1.0},
        {3.0, 4.0}
    };

    public TriangleClassifier() {
        buildModel();
    }

    protected void buildModel() {
        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .seed(modelSeed)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Adam(0.1))
                .l2(0.001)
                .list()
                .layer(new DenseLayer.Builder()
                        .nIn(2)
                        .nOut(2)
                        .activation(Activation.TANH)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.XENT)
                        .nIn(2)
                        .nOut(1)
                        .activation(Activation.SIGMOID)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .build();

        model = new MultiLayerNetwork(config);
        model.init();
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

    public DataSet generateDataSet(int numSamples, long seed) {
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

    public void train(int numEpochs, int numSamples) {
        DataSet trainingData = generateDataSet(numSamples, trainSeed);
        
        System.out.println("Начало обучения классификатора треугольников...");
        System.out.println("Вершины треугольника: " + 
                         Arrays.deepToString(triangleVertices));
        
        for (int epoch = 0; epoch < numEpochs; epoch++) {
            model.fit(trainingData);
            
            if (epoch % 50 == 0) {
                double score = model.score();
                System.out.printf("Эпоха %d, Loss: %.4f%n", epoch, score);
            }
        }
        System.out.println("Обучение завершено!");
    }

    public boolean predict(double x, double y) {
        INDArray input = Nd4j.create(new double[][]{{x, y}});
        INDArray output = model.output(input);
        return output.getDouble(0) > 0.5;
    }

    public double predictProbability(double x, double y) {
        INDArray input = Nd4j.create(new double[][]{{x, y}});
        INDArray output = model.output(input);
        return output.getDouble(0);
    }

    public void evaluate(int numTestSamples) {
        DataSet testData = generateDataSet(numTestSamples, validationSeed);
        INDArray predictions = model.output(testData.getFeatures());
        
        int correct = 0;
        for (int i = 0; i < numTestSamples; i++) {
            double predictedProbability = predictions.getDouble(i, 0);
            boolean predictedInside = predictedProbability > 0.5;
            double actualLabel = testData.getLabels().getDouble(i, 0);
            boolean actualInside = actualLabel > 0.5;
            
            if (predictedInside == actualInside) {
                correct++;
            }
        }
        
        double accuracy = correct * 100.0 / numTestSamples;
        System.out.printf("Точность на тестовых данных: %d/%d (%.2f%%)%n", 
                         correct, numTestSamples, accuracy);
    }

    public void testExamples() {
        double[][] testPoints = {
            {3.0, 2.0},   // Центр треугольника -> внутри
            {1.0, 1.5},   // Вершина A -> внутри
            {4.0, 1.5},   // Вершина B -> внутри  
            {3.0, 3.0},   // Вершина C -> внутри
            {2.5, 2.0},   // Внутри треугольника
            {0.5, 0.5},   // Снаружи
            {5.5, 5.5},   // Снаружи
            {3.0, 0.5},   // Снаружи (под треугольником)
            {0.0, 0.0},   // Снаружи
            {6.0, 6.0}    // Снаружи
        };
        
        System.out.println("\n=== Тестирование на примерах ===");
        System.out.println("Точка (x, y) -> Ожидаемый класс | Предсказанный класс | Вероятность");
        System.out.println("-------------------------------------------------------------------");
        
        for (double[] point : testPoints) {
            double x = point[0], y = point[1];
            boolean expectedInside = isPointInTriangle(x, y);
            boolean predictedInside = predict(x, y);
            double probability = predictProbability(x, y);
            
            String expectedClass = expectedInside ? "Внутри" : "Снаружи";
            String predictedClassStr = predictedInside ? "Внутри" : "Снаружи";
            
            System.out.printf("(%.1f, %.1f) -> %s | %s", x, y, expectedClass, predictedClassStr);
            System.out.printf(" (P=%.3f)%n", probability);
        }
    }

    public void saveToCSV(DataSet dataset, String filename) {
        try (PrintWriter writer = new PrintWriter(new FileWriter(filename))) {
            writer.println("x,y,class,class_name");
            
            INDArray features = dataset.getFeatures();
            INDArray labels = dataset.getLabels();
            
            for (int i = 0; i < features.rows(); i++) {
                double x = features.getDouble(i, 0);
                double y = features.getDouble(i, 1);
                double label = labels.getDouble(i, 0);
                int classIndex = label > 0.5 ? 0 : 1; // 0 = внутри, 1 = снаружи
                String className = classIndex == 0 ? "inside" : "outside";
                
                writer.printf("%.6f,%.6f,%d,%s%n", x, y, classIndex, className);
            }
            
            System.out.println("Датасет треугольников сохранен в: " + filename);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void generateAndSaveDataset(int numSamples, long seed, String filename) {
        DataSet dataset = generateDataSet(numSamples, seed);
        saveToCSV(dataset, filename);
    }


    public void printAllWeightsAndBiases() {
        int numLayers = model.getnLayers();
        
        System.out.println("\n=== ВЕСА И СМЕЩЕНИЯ МОДЕЛИ ===");
        
        for (int layerIndex = 0; layerIndex < numLayers; layerIndex++) {
            System.out.printf("\n--- Слой %d ---%n", layerIndex);
            
            // Получаем веса слоя
            INDArray weights = model.getLayer(layerIndex).getParam("W");
            double[][] weightsArray = new double[weights.rows()][weights.columns()];
            
            for (int i = 0; i < weights.rows(); i++) {
                for (int j = 0; j < weights.columns(); j++) {
                    weightsArray[i][j] = weights.getDouble(i, j);
                }
            }
            
            System.out.println("Веса (" + weights.rows() + "x" + weights.columns() + "):");
            for (double[] row : weightsArray) {
                System.out.println(Arrays.toString(row));
            }
            
            // Получаем смещения слоя
            INDArray biases = model.getLayer(layerIndex).getParam("b");
            double[] biasesArray = new double[(int) biases.length()];
            
            for (int i = 0; i < biases.length(); i++) {
                biasesArray[i] = biases.getDouble(i);
            }
            
            System.out.println("Смещения (" + biasesArray.length + "):");
            System.out.println(Arrays.toString(biasesArray));
        }
    }

    public static void main(String[] args) {
        TriangleClassifier classifier = new TriangleClassifier();
        
        classifier.generateAndSaveDataset(1000, trainSeed, "triangle_dataset.csv");
        
        classifier.train(1000, 10000);
        
        classifier.evaluate(1000);
        
        classifier.testExamples();

        classifier.printAllWeightsAndBiases();
        
        System.out.println("\n=== Информация о треугольнике ===");
        System.out.println("Вершины: A" + Arrays.toString(classifier.triangleVertices[0]) + 
                         ", B" + Arrays.toString(classifier.triangleVertices[1]) + 
                         ", C" + Arrays.toString(classifier.triangleVertices[2]));
    }
}