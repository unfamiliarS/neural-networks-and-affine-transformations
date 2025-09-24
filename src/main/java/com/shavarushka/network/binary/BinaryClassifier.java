package com.shavarushka.network.binary;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Arrays;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class BinaryClassifier {

    private MultiLayerNetwork model;

    private static final int modelSeed = 67890;
    public static final int trainSeed = 12345;
    public static final int validationSeed = 12345;

    public BinaryClassifier() {
        buildModel();
    }

    private void buildModel() {
        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .seed(modelSeed)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Adam(0.1))
                .l2(1e-4)
                .list()
                .layer(new DenseLayer.Builder()
                        .nIn(2)
                        .nOut(3)
                        .activation(Activation.RELU)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.XENT)
                        .nIn(3)
                        .nOut(1)
                        .activation(Activation.SIGMOID)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                // .layer(new OutputLayer.Builder(LossFunctions.LossFunction.XENT)
                //         .nIn(2)
                //         .nOut(1)
                //         .activation(Activation.SIGMOID)
                //         .weightInit(WeightInit.XAVIER)
                //         .build())
                .build();

        model = new MultiLayerNetwork(config);
        model.init();
    }

    public DataSet generateDataSet(int numSamples, long seed) {
        Nd4j.getRandom().setSeed(seed);
        INDArray features = Nd4j.randn(numSamples, 2);
        
        // class = 1 if x1 + x2 >= 0
        INDArray labels = Nd4j.create(numSamples, 1);
        for (int i = 0; i < numSamples; i++) {
            double x1 = features.getDouble(i, 0);
            double x2 = features.getDouble(i, 1);
            double label = (x1 + x2 >= 0) ? 1.0 : 0.0;
            labels.putScalar(i, 0, label);
        }
        
        return new DataSet(features, labels);
    }

    public void saveDatasetToCSV(DataSet dataset, String filename) {
        try (PrintWriter writer = new PrintWriter(new FileWriter(filename))) {
            writer.println("x1,x2,label");
            
            INDArray features = dataset.getFeatures();
            INDArray labels = dataset.getLabels();
            
            for (int i = 0; i < features.rows(); i++) {
                double x1 = features.getDouble(i, 0);
                double x2 = features.getDouble(i, 1);
                double label = labels.getDouble(i, 0);
                writer.printf("%.6f,%.6f,%.0f%n", x1, x2, label);
            }
            
            System.out.println("Датасет сохранен в: " + filename);
        } catch (IOException e) {
            System.err.println("Ошибка при сохранении датасета: " + e.getMessage());
        }
    }

    public void train(int numEpochs, int numSamples) {
        DataSet trainingData = generateDataSet(numSamples, trainSeed);
        
        for (int epoch = 0; epoch < numEpochs; epoch++) {
            model.fit(trainingData);
            
            System.out.println("Epoch " + epoch + ", Loss: " + model.score());
        }
    }
    
    public void evaluate(int testSamples) {
        DataSet testData = generateDataSet(testSamples, validationSeed);
        Evaluation eval = new Evaluation();
        
        INDArray output = model.output(testData.getFeatures());
        eval.eval(testData.getLabels(), output);
        
        System.out.println("\n=== Evaluation Results ===");
        System.out.println(eval.stats());
    }

    public double predict(double x1, double x2) {
        INDArray input = Nd4j.create(new double[][]{{x1, x2}});
        INDArray output = model.output(input);
        return output.getDouble(0);
    }

    public void saveModel(String filePath) throws IOException {
        File modelFile = new File(filePath);
        ModelSerializer.writeModel(model, modelFile, true);
        System.out.println("Model saved to: " + filePath);
    }

    public void loadModel(String filePath) throws IOException {
        File modelFile = new File(filePath);
        model = ModelSerializer.restoreMultiLayerNetwork(modelFile);
        System.out.println("Model loaded from: " + filePath);
    }

    /**
     * Get weights for a specific layer as a 2D double array.
     * @param layerIndex Index of the layer (0-based).
     * @return Weights matrix for the specified layer as a 2D double array.
     */
    public double[][] getWeightsAsArray(int layerIndex) {
        INDArray weights = model.getLayer(layerIndex).getParam("W");
        double[][] weightsArray = new double[weights.rows()][weights.columns()];
        for (int i = 0; i < weights.rows(); i++) {
            for (int j = 0; j < weights.columns(); j++) {
                weightsArray[i][j] = weights.getDouble(i, j);
            }
        }
        return weightsArray;
    }

    /**
     * Get biases for a specific layer as a double array.
     * @param layerIndex Index of the layer (0-based).
     * @return Biases vector for the specified layer as a double array.
     */
    public double[] getBiasesAsArray(int layerIndex) {
        INDArray biases = model.getLayer(layerIndex).getParam("b");
        double[] biasesArray = new double[(int) biases.length()];
        for (int i = 0; i < biases.length(); i++) {
            biasesArray[i] = biases.getDouble(i);
        }
        return biasesArray;
    }

    /**
     * Print weights and biases for all layers as arrays.
     */
    public void printAllWeightsAndBiasesAsArrays() {
        int numLayers = model.getnLayers();
        for (int i = 0; i < numLayers; i++) {
            System.out.println("Layer " + i + " weights:");
            double[][] weightsArray = getWeightsAsArray(i);
            for (double[] row : weightsArray) {
                System.out.println(Arrays.toString(row));
            }

            System.out.println("Layer " + i + " biases:");
            double[] biasesArray = getBiasesAsArray(i);
            System.out.println(Arrays.toString(biasesArray));
        }
    }
}
