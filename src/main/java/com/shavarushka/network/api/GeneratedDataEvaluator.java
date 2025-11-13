package com.shavarushka.network.api;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.evaluation.classification.Evaluation;

public class GeneratedDataEvaluator extends ModelEvaluator {

    private DataGenerator dataGenerator;

    public GeneratedDataEvaluator(MultiLayerNetwork model, DataGenerator dataGenerator) {
        super(model);
        this.dataGenerator = dataGenerator;
    }

    @Override
    public Evaluation evaluate() {
        DataSet testData = dataGenerator.generate();
        INDArray output = model.output(testData.getFeatures());
        Evaluation evaluation = new Evaluation();
        evaluation.eval(testData.getLabels(), output);
        return evaluation;
    }
}
