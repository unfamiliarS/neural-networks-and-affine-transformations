package com.shavarushka.network;

import com.shavarushka.network.api.BaseEvaluator;
import com.shavarushka.network.api.BaseNetwork;
import com.shavarushka.network.api.DataIterators;
import com.shavarushka.network.api.Trainer;

import java.io.File;
import java.io.IOException;

public class MNISTClassifier {
    private BaseNetwork network;
    private Trainer trainer;
    private BaseEvaluator evaluator;
    private MNISTPredictor predictor;
    private WeightManager weightManager;

    private MNISTClassifier(BaseNetwork network, DataIterators dataIterators) {
        this.network = network;
        this.evaluator = new MNISTEvaluator(network.getModel(), dataIterators);
        this.trainer = new MNISTTrainer(network.getModel(), dataIterators, evaluator);
        this.predictor = new MNISTPredictor(network.getModel());
        this.weightManager = new WeightManager(network.getModel());
    }

    public static MNISTClassifier create() {
        BaseNetwork model = MNISTNetwork.create();
        DataIterators dataManager = new MNISTDataIterators();
        return new MNISTClassifier(model, dataManager);
    }

    public static MNISTClassifier load(String filePath) {
        try {
            BaseNetwork model = MNISTNetwork.load(filePath);
            DataIterators dataManager = new MNISTDataIterators();
            return new MNISTClassifier(model, dataManager);
        } catch (Exception e) {
            System.err.println("Failed to load model: " + e.getMessage());
            return null;
        }
    }

    public void save(String filePath) {
        try {
            network.save(filePath);
        } catch (IOException e) {
            System.err.println("Failed to save model: " + e.getMessage());
        }
    }

    public void train(int numEpochs) {
        trainer.train(numEpochs);
    }

    public void evaluate() {
        evaluator.printEvaluation();
    }

    public void printInfo() {
        System.out.println("Model architecture:");
        System.out.println(network.getSummary());
        
        System.out.println("\nLayer parameters:");
        for (int i = 0; i < network.getNumLayers(); i++) {
            System.out.println("Layer " + i + ": " + 
                    weightManager.getLayerParameterCount(i) + " parameters");
        }
    }

    public void predict(String imagePath) {
        try {
            File imageFile = new File(imagePath);
            MNISTPredictor.PredictionResult result = predictor.predictDetailedFromImage(imageFile);
            result.printDetails();
        } catch (IOException e) {
            System.err.println("Error loading image: " + e.getMessage());
        }
    }

    public double[][] getWeights(int layerIndex) {
        return weightManager.getLayerWeights(layerIndex);
    }
    
    public double[][][] getWeights() {
        return weightManager.getAllWeights();
    }

    public void setWeights(int layerIndex, double[][] newWeights) {
        weightManager.setLayerWeights(layerIndex, newWeights);
    }

    public void setWeights(double[][][] allWeights) {
        weightManager.setAllWeights(allWeights);
    }

    public void printWeights() {
        weightManager.printWeights();
    }
}
