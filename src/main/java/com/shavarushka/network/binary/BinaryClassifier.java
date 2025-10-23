/* package com.shavarushka.network.binary;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;

import com.shavarushka.network.ModelEvaluator;
import com.shavarushka.network.ModelTrainer;
import com.shavarushka.network.WeightManager;
import com.shavarushka.network.api.Classifier;
import com.shavarushka.network.api.DataIterators;
import com.shavarushka.network.api.Network;

public abstract class BinaryClassifier implements Classifier {

    protected Network networkModel;
    protected ModelEvaluator evaluator;
    protected ModelTrainer trainer;
    protected WeightManager weightManager;

    public BinaryClassifier(Network networkModel, DataIterators dataIterators) {
        this.networkModel = networkModel;
        this.evaluator = new ModelEvaluator(networkModel.getModel(), dataIterators);
        this.trainer = new ModelTrainer(networkModel.getModel(), dataIterators, evaluator);
        this.weightManager = new WeightManager(networkModel.getModel());
    }

    public abstract double[] predict(double x1, double x2);

    public void train(int numEpochs) {
        for (int epoch = 0; epoch < numEpochs; epoch++) {
            model.fit(trainingData);
            
            if (epoch % 10 == 0)
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

    public double[] getBiasesAsArray(int layerIndex) {
        INDArray biases = model.getLayer(layerIndex).getParam("b");
        double[] biasesArray = new double[(int) biases.length()];
        for (int i = 0; i < biases.length(); i++) {
            biasesArray[i] = biases.getDouble(i);
        }
        return biasesArray;
    }

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
 */