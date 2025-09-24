package com.shavarushka.network;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.File;
import java.io.IOException;

public class MNISTClassifier {

    private MultiLayerNetwork model;
    private DataSetIterator mnistTrain;
    private DataSetIterator mnistTest;

    private final int numClasses = 10;
    private final int seed = 12345;
    private final int batchSize = 64;

    public MNISTClassifier() {
        buildModel();
    }

    private void buildModel() {
        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Adam(0.001))
                .l2(1e-4)
                .list()
                .layer(new DenseLayer.Builder()
                        .nIn(28 * 28)
                        .nOut(512)
                        .activation(Activation.RELU)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .layer(new DenseLayer.Builder()
                        .nIn(512)
                        .nOut(256)
                        .activation(Activation.RELU)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .layer(new DenseLayer.Builder()
                        .nIn(256)
                        .nOut(128)
                        .activation(Activation.RELU)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nIn(128)
                        .nOut(numClasses)
                        .activation(Activation.SOFTMAX)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .build();

        model = new MultiLayerNetwork(config);
        model.init();
        model.setListeners(new ScoreIterationListener(100));
    }

    public void train(int numEpochs) throws IOException {
        mnistTrain = new MnistDataSetIterator(batchSize, true, seed);
        mnistTest = new MnistDataSetIterator(batchSize, false, seed);
        
        System.out.println("Starting MNIST training...");
        System.out.println("Batch size: " + batchSize);
        System.out.println("Number of epochs: " + numEpochs);
        
        for (int epoch = 1; epoch <= numEpochs; epoch++) {
            System.out.println("Epoch: " + epoch + "/" + numEpochs);
            model.fit(mnistTrain);
            mnistTrain.reset();

            evaluateAccuracy(epoch);
        }
        
        System.out.println("\nFinal evaluation on test data:");
        fullEvaluation();
    }

    private void evaluateAccuracy(int epoch) {
        Evaluation eval = model.evaluate(mnistTest);
        System.out.println("Accuracy: " + String.format("%.4f", eval.accuracy()));
        mnistTest.reset();
    }

    public void fullEvaluation() throws IOException {
        Evaluation evaluation = model.evaluate(mnistTest);
        System.out.println(evaluation.stats());
        System.out.println("Confusion Matrix:\n" + evaluation.confusionToString());
    }

    public void testSinglePrediction() throws IOException {
        DataSetIterator testIter = new MnistDataSetIterator(1, false, seed);
        DataSet testData = testIter.next();
        
        INDArray features = testData.getFeatures();
        INDArray labels = testData.getLabels();
        
        int actualLabel = labels.argMax(1).getInt(0);
        INDArray prediction = model.output(features);
        int predictedLabel = prediction.argMax(1).getInt(0);
        
        System.out.println("Test prediction:");
        System.out.println("Actual digit: " + actualLabel);
        System.out.println("Predicted digit: " + predictedLabel);
        System.out.println("Probabilities: " + prediction);
        System.out.println("Correct: " + (actualLabel == predictedLabel));
    }

    public void saveModel(String filePath) throws IOException {
        ModelSerializer.writeModel(model, new File(filePath), true);
        System.out.println("Model saved: " + filePath);    
    }

    public void loadModel(String filePath) throws IOException {
        model = ModelSerializer.restoreMultiLayerNetwork(new File(filePath));
        System.out.println("Model loaded: " + filePath);
    }

    public void printModelInfo() {
        System.out.println("Model architecture:");
        System.out.println(model.summary());
        
        System.out.println("\nLayer parameters:");
        for (int i = 0; i < model.getLayers().length; i++) {
            System.out.println("Layer " + i + ": " + 
                    model.getLayer(i).numParams() + " parameters");
        }
    }

    public double[] getWeightsAsArray(int layerIndex) {
        INDArray weights = model.getLayer(layerIndex).getParam("W");
        return weights.data().asDouble();
    }

    public double[][] getWeightsAsMatrix(int layerIndex) {
        INDArray weights = model.getLayer(layerIndex).getParam("W");
        double[][] matrix = new double[weights.rows()][weights.columns()];
        
        for (int i = 0; i < weights.rows(); i++) {
            for (int j = 0; j < weights.columns(); j++) {
                matrix[i][j] = weights.getDouble(i, j);
            }
        }
        return matrix;
    }

    public static void printWeightMatrix(double[][] matrix) {
        for (int i = 0; i < 3; i++) {
            System.out.println("i: " + i);
            for (int j = 0; j < 5; j++ ) {
                System.out.println(matrix[i][j]);
            }
        }
    }

    public static void main(String[] args) {
        try {
            MNISTClassifier classifier = new MNISTClassifier();
            
            // classifier.train(6);
            
            // classifier.testSinglePrediction();
            
            // classifier.saveModel("mnist_model.dl4j");
            
            // classifier.printModelInfo();
            
            classifier.loadModel("mnist_model.dl4j");
            double[][] matrix = classifier.getWeightsAsMatrix(3);
            printWeightMatrix(matrix);

        } catch (Exception e) {
            System.err.println("Error: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
