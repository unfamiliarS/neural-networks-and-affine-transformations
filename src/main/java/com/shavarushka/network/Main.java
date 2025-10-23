package com.shavarushka.network;

public class Main {
    public static void main(String[] args) {
        MNISTClassifier classifier = MNISTClassifier.create();
        classifier.train(16);
        classifier.evaluate();

        classifier.save("mnist-model.zip");

        classifier.printInfo();
    }
}
