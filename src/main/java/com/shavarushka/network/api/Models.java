package com.shavarushka.network.api;

public enum Models {
    MNIST("src/main/resources/mnist-model.zip", "", false),
    SIMPLE_MNIST("src/main/resources/simple-mnist.zip", "", false),
    TRIANGLE("src/main/resources/triangle-classifier.zip", "src/main/python/triangle/dataset.csv", true),
    TWO_TRIANGLES("src/main/resources/two-triangle.zip", "src/main/python/multipletriangle/dataset.csv", true);

    private String modelPath;
    private String datasetPath;
    private boolean isVisualizable;
    
    public String getModelPath() {
        return modelPath;
    }
    
    public String getDatasetPath() {
        return datasetPath;
    }

    public boolean isVisualizable() {
        return isVisualizable;
    }

    Models(String modelPath, String datasetPath, boolean isVisualizable) {
        this.modelPath = modelPath;
        this.datasetPath = datasetPath;
        this.isVisualizable = isVisualizable;
    }

    public static Models get(String modelStr) {
        return Models.valueOf(modelStr.toUpperCase().replace("-", "_"));
    }
}
