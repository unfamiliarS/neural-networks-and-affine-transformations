package com.shavarushka.network.api;

import java.io.File;

public interface ModelImagePredictor extends ModelPredictor {
    PredictionResult predictFromImage(File imageFile);
}
