package com.shavarushka.network.api;

import java.io.File;
import java.io.IOException;

public interface ModelImagePredictorExtended extends ModelImagePredictor {
    PredictionResult predictDetailed(double[][] imageData);
    PredictionResult predictDetailedFromImage(File imageFile) throws IOException;
}
