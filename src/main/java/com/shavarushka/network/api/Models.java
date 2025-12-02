package com.shavarushka.network.api;

public enum Models {
    MNIST(System.getenv("HOME") + "/.local/affine/mnist/mnist-model.zip", "", false),
    SIMPLE_MNIST(System.getenv("HOME") + "/.local/affine/simple-mnist/simple-mnist.zip", "", false),
    TRIANGLE(System.getenv("HOME") + "/.local/affine/triangle/triangle-classifier.zip", System.getenv("HOME") + "/.local/affine/triangle/dataset.csv", true),
    TWO_TRIANGLES(System.getenv("HOME") + "/.local/affine/two-triangles/two-triangle.zip", System.getenv("HOME") + "/.local/affine/two-triangles/dataset.csv", true);

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
