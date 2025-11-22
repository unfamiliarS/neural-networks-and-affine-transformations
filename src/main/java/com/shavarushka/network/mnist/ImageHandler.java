package com.shavarushka.network.mnist;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

import javax.imageio.ImageIO;

public class ImageHandler {

    public static double[][] load(File imageFile) {
        try {
            BufferedImage image = ImageIO.read(imageFile);
            double[][] imageData = new double[28][28];
            for (int y = 0; y < 28; y++) {
                for (int x = 0; x < 28; x++) {
                    int rgb = image.getRGB(x, y);
                    int gray = (rgb >> 16) & 0xFF;
                    imageData[y][x] = gray;
                }
            }

            return imageData;

        } catch (IOException e) {
            e.printStackTrace();
        }
        return null;
    }

    public static double[] flattenImage(double[][] image) {
        int height = image.length;
        int width = image[0].length;
        double[] flattened = new double[height*width];
        int index = 0;

        for (int i = 0; i < height; i++)
            for (int j = 0; j < width; j++)
                flattened[index++] = image[i][j];

        return flattened;
    }
}
