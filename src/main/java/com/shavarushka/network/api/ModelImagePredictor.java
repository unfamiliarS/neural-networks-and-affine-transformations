package com.shavarushka.network.api;

import java.io.File;
import java.io.IOException;

public interface ModelImagePredictor {
    double[] predict(double[][] imageData);
    int predictDigit(double[][] imageData);
    int predictFromImage(File imageFile) throws IOException;
}
