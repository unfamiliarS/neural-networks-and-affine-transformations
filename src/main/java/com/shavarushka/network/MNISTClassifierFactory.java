package com.shavarushka.network;

import com.shavarushka.network.api.DataIterators;
import com.shavarushka.network.api.Network;

public class MNISTClassifierFactory {
    
    public static MNISTClassifier createNew() {
        Network model = FullyConnectedMNISTNetwork.create();
        DataIterators dataManager = new MNISTDataIterators();
        return new MNISTClassifier(model, dataManager);
    }

    public static MNISTClassifier loadFromFile(String filePath) {
        try {
            Network model = FullyConnectedMNISTNetwork.load(filePath);
            DataIterators dataManager = new MNISTDataIterators();
            return new MNISTClassifier(model, dataManager);
        } catch (Exception e) {
            System.err.println("Failed to load model: " + e.getMessage());
            return null;
        }
    }
}
