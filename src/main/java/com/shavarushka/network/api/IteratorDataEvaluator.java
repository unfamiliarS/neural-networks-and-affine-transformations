package com.shavarushka.network.api;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.evaluation.classification.Evaluation;

public class IteratorDataEvaluator extends ModelEvaluator {

    private DataIterators dataIterators;

    public IteratorDataEvaluator(MultiLayerNetwork model, DataIterators dataIterators) {
        super(model);
        this.dataIterators = dataIterators;
    }

    @Override
    public Evaluation evaluate() {
        DataSetIterator testIterator = dataIterators.getTestIterator();
        Evaluation evaluation = model.evaluate(testIterator);
        dataIterators.resetTestIterator();
        return evaluation;
    }
}
