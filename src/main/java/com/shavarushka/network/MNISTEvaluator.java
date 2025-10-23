package com.shavarushka.network;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import com.shavarushka.network.api.BaseEvaluator;
import com.shavarushka.network.api.DataIterators;

import org.nd4j.evaluation.classification.Evaluation;

public class MNISTEvaluator extends BaseEvaluator {

    private DataIterators dataIterators;

    public MNISTEvaluator(MultiLayerNetwork model, DataIterators dataIterators) {
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
