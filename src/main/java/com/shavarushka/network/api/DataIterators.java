package com.shavarushka.network.api;

import java.io.IOException;

import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

public abstract class DataIterators {

    protected DataSetIterator trainIterator;
    protected DataSetIterator testIterator;

    public DataIterators() {
        initializeIterators();
    }

    private void initializeIterators() {
        try {
            initializeTrainIterator();
            initializeTestIterator();
        } catch (IOException e) {
            throw new RuntimeException("Failed to initialize MNIST dataset iterators", e);
        }
    }

    protected abstract void initializeTrainIterator() throws IOException;
    protected abstract void initializeTestIterator() throws IOException;

    public DataSetIterator getTrainIterator() {
        return trainIterator;
    }

    public DataSetIterator getTestIterator() {
        return testIterator;
    }

    public void resetTrainIterator() {
        trainIterator.reset();
    }

    public void resetTestIterator() {
        testIterator.reset();
    }

    public abstract int getBatchSize();
}
