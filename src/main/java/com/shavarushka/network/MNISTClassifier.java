package com.shavarushka.network;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
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
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import com.shavarushka.network.api.Classifier;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;

public class MNISTClassifier implements Classifier {

    private MultiLayerNetwork model;
    private DataSetIterator mnistTrain;
    private DataSetIterator mnistTest;

    private static final int numClasses = 10;
    private static final int seed = 12345;
    private static final int batchSize = 128;

    private MNISTClassifier(MultiLayerNetwork model) {
        this.model = model;
        this.model.init();
    }

    public static MNISTClassifier build() {
        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Adam(0.001))
                .l2(1e-4)
                .list()
                .layer(new DenseLayer.Builder()
                        .nIn(28 * 28)
                        .nOut(64)
                        .activation(Activation.RELU)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .layer(new DenseLayer.Builder()
                        .nIn(64)
                        .nOut(32)
                        .activation(Activation.RELU)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .layer(new DenseLayer.Builder()
                        .nIn(32)
                        .nOut(16)
                        .activation(Activation.RELU)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nIn(16)
                        .nOut(numClasses)
                        .activation(Activation.SOFTMAX)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(config);
        return new MNISTClassifier(model);
    }

    public static MNISTClassifier load(String filePath) {
        try {
            MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(new File(filePath));
            System.out.println("Model loaded: " + filePath);
            return new MNISTClassifier(model);
        } catch (IOException e) {
            e.printStackTrace();
        }
        return null;
    }

    @Override
    public void train(int numEpochs) {
        initDatasetIterators();

        System.out.println("Starting MNIST training...");
        System.out.println("Batch size: " + batchSize);
        System.out.println("Number of epochs: " + numEpochs);
        
        for (int epoch = 1; epoch <= numEpochs; epoch++) {
            System.out.println("Epoch: " + epoch + "/" + numEpochs);
            model.fit(mnistTrain);
            mnistTrain.reset();
            accuracy();
        }   
    }

    private void initDatasetIterators() {
        try {
            mnistTrain = new MnistDataSetIterator(batchSize, true, seed);
            mnistTest = new MnistDataSetIterator(batchSize, false, seed);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void accuracy() {
        Evaluation eval = model.evaluate(mnistTest);
        System.out.println("Accuracy: " + String.format("%.4f", eval.accuracy()));
        mnistTest.reset();
    }

    @Override
    public void evaluate() {
        if (mnistTest == null)
            initDatasetIterators();

        Evaluation evaluation = model.evaluate(mnistTest);
        System.out.println(evaluation.stats());
        System.out.println("Confusion Matrix:\n" + evaluation.confusionToString());
        mnistTest.reset();
    }

    @Override
    public void save(String filePath) {
        try {
            ModelSerializer.writeModel(model, new File(filePath), true);
        } catch (IOException e) {
            e.printStackTrace();
        }
        System.out.println("Model saved: " + filePath);    
    }

    public void printInfo() {
        System.out.println("Model architecture:");
        System.out.println(model.summary());
        
        System.out.println("\nLayer parameters:");
        for (int i = 0; i < model.getLayers().length; i++) {
            System.out.println("Layer " + i + ": " + 
                    model.getLayer(i).numParams() + " parameters");
        }
    }

    public double[][] getWeights(int layerIndex) {
        INDArray weights = model.getLayer(layerIndex).getParam("W");
        double[][] weightsArray = new double[weights.rows()][weights.columns()];
        for (int i = 0; i < weights.rows(); i++) {
            for (int j = 0; j < weights.columns(); j++) {
                weightsArray[i][j] = weights.getDouble(i, j);
            }
        }

        return weightsArray;
    }
    
    public double[][][] getWeights() {
        int numLayers = model.getnLayers();
        double[][][] weights = new double[numLayers][][];
        for (int i = 0; i < numLayers; i++)
            weights[i] = getWeights(i);

        return weights;
    }

    public void setWeights(int layerIndex, double[][] newWeights) {
        INDArray weights = model.getLayer(layerIndex).getParam("W");
        INDArray newWeightsArray = Nd4j.create(newWeights).reshape(weights.shape());
        model.getLayer(layerIndex).setParam("W", newWeightsArray);
    }

    public void setWeights(double[][][] allWeights) {
        for (int i = 0; i < allWeights.length; i++)
            setWeights(i, allWeights[i]);
    }

    public void printWeights() {
        double[][][] matrix = getWeights();

        for (int i = 0; i < matrix.length; i++) {
            System.out.println("Layer " + i + ":");
            double[][] weightsArray = getWeights(i);
            for (double[] row : weightsArray)
                System.out.println(Arrays.toString(row));
        }
    }

    public static void main(String[] args) {
        var classifier = MNISTClassifier.load("mnist-model.zip");

        classifier.evaluate();

        double[][] firstWeights = classifier.getWeights(0);
        firstWeights[0][0] = 0.0;
        classifier.setWeights(0, firstWeights);

        classifier.evaluate();
    }
}
