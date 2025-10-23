package com.shavarushka.network;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;

import com.shavarushka.network.api.DataIterators;

import java.io.IOException;

public class MNISTDataIterators extends DataIterators {

    private static final int batchSize = 128;
    private static final int seed = 12345;

    public MNISTDataIterators() {
        super();
    }

    @Override
    protected void initializeTrainIterator() throws IOException {
        trainIterator = new MnistDataSetIterator(batchSize, true, seed);
    }

    @Override
    protected void initializeTestIterator() throws IOException {
        testIterator = new MnistDataSetIterator(batchSize, false, seed);
    }

    public int getBatchSize() {
        return batchSize;
    }
}
