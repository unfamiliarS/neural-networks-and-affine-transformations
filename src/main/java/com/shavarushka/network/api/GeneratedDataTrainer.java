package com.shavarushka.network.api;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.dataset.DataSet;

public class GeneratedDataTrainer extends ModelTrainer {

    private DataGenerator dataGenerator;

    public GeneratedDataTrainer(MultiLayerNetwork model, DataGenerator dataGenerator, ModelEvaluator evaluator) {
        super(model, evaluator);
        this.dataGenerator = dataGenerator;
    }

    @Override
    public void train(int numEpochs) {
        DataSet trainingData = dataGenerator.generate();

        System.out.println("Starting training...");
        for (int epoch = 1; epoch <= numEpochs; epoch++) {
            System.out.println("Epoch: " + epoch + "/" + numEpochs);
            model.fit(trainingData);
            evaluator.printAccuracy();
        }
    }

}
