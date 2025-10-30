package com.shavarushka.network;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import com.shavarushka.network.api.ModelEvaluator;
import com.shavarushka.network.api.ModelTrainer;
import com.shavarushka.network.api.DataIterators;

public class MNISTTrainer extends ModelTrainer {

    private DataIterators dataIterators;

    public MNISTTrainer(MultiLayerNetwork model, DataIterators dataIterators, ModelEvaluator evaluator) {
        super(model, evaluator);
        this.dataIterators = dataIterators;
    }

    @Override
    public void train(int numEpochs) {
        DataSetIterator trainIterator = dataIterators.getTrainIterator();
        
        System.out.println("Starting MNIST training...");
        System.out.println("Batch size: " + dataIterators.getBatchSize());
        System.out.println("Number of epochs: " + numEpochs);
        
        for (int epoch = 1; epoch <= numEpochs; epoch++) {
            System.out.println("Epoch: " + epoch + "/" + numEpochs);
            model.fit(trainIterator);
            dataIterators.resetTrainIterator();
            evaluator.printAccuracy();
        }   
    }
}
