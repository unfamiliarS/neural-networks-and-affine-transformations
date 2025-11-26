package com.shavarushka.cli.commands;

import java.io.File;

import com.shavarushka.network.mnist.ImageHandler;

class ImageDataExtraction implements DataExtractionStrategy {

    String imagePath;

    public ImageDataExtraction(String data) {
        imagePath = data;
    }

    @Override
    public double[] extract() {
        double[][] imageData = ImageHandler.load(new File(imagePath));
        return ImageHandler.flattenImage(imageData);
    }
}
