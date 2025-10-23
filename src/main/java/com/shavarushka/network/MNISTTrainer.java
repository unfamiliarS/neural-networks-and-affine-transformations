package com.shavarushka.network;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import com.shavarushka.network.api.BaseEvaluator;
import com.shavarushka.network.api.DataIterators;
import com.shavarushka.network.api.Trainer;

public class MNISTTrainer implements Trainer {
    private MultiLayerNetwork model;
    private DataIterators dataManager;
    private BaseEvaluator evaluator;

    public MNISTTrainer(MultiLayerNetwork model, DataIterators dataManager, BaseEvaluator evaluator) {
        this.model = model;
        this.dataManager = dataManager;
        this.evaluator = evaluator;
    }

    public void train(int numEpochs) {
        DataSetIterator trainIterator = dataManager.getTrainIterator();
        
        System.out.println("Starting MNIST training...");
        System.out.println("Batch size: " + dataManager.getBatchSize());
        System.out.println("Number of epochs: " + numEpochs);
        
        for (int epoch = 1; epoch <= numEpochs; epoch++) {
            System.out.println("Epoch: " + epoch + "/" + numEpochs);
            model.fit(trainIterator);
            dataManager.resetTrainIterator();
            evaluator.printAccuracy();
        }   
    }
}
