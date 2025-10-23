package com.shavarushka.network;

import com.shavarushka.network.api.Classifier;
import com.shavarushka.network.api.DataIterators;
import com.shavarushka.network.api.Network;

import java.io.IOException;

public class MNISTClassifier implements Classifier {
    private Network networkModel;
    private ModelEvaluator evaluator;
    private ModelTrainer trainer;
    private WeightManager weightManager;

    public MNISTClassifier(Network networkModel, DataIterators dataIterators) {
        this.networkModel = networkModel;
        this.evaluator = new ModelEvaluator(networkModel.getModel(), dataIterators);
        this.trainer = new ModelTrainer(networkModel.getModel(), dataIterators, evaluator);
        this.weightManager = new WeightManager(networkModel.getModel());
    }

    @Override
    public void train(int numEpochs) {
        trainer.train(numEpochs);
    }

    @Override
    public void evaluate() {
        evaluator.printEvaluation();
    }

    @Override
    public void save(String filePath) {
        try {
            networkModel.save(filePath);
        } catch (IOException e) {
            System.err.println("Failed to save model: " + e.getMessage());
        }
    }

    public void printInfo() {
        System.out.println("Model architecture:");
        System.out.println(networkModel.getSummary());
        
        System.out.println("\nLayer parameters:");
        for (int i = 0; i < networkModel.getNumLayers(); i++) {
            System.out.println("Layer " + i + ": " + 
                    weightManager.getLayerParameterCount(i) + " parameters");
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
