package com.shavarushka.cli.commands;

import java.io.File;
import java.util.Map;

import com.shavarushka.affine.AffineTransformations;
import com.shavarushka.affine.MatrixUtils;
import com.shavarushka.network.api.ModelLoader;
import com.shavarushka.network.api.ModelPredictor;
import com.shavarushka.network.api.WeightsManager;
import com.shavarushka.network.api.fabric.ModelFabric;
import com.shavarushka.network.mnist.ImageHandler;

public class RotationCommand implements Command {

    private double rotationAngle;
    private Map<String,String> requiredArgs;
    private ModelFabric fabric;

    public RotationCommand(Map<String,String> requiredArgs, String rotationAngle) {
        this.requiredArgs = requiredArgs;
        this.rotationAngle = Double.parseDouble(rotationAngle);

        fabric = ModelFabric.createFabric(
            requiredArgs.get("mtype"),
            ModelLoader.load(requiredArgs.get("model"))
        );
    }

    @Override
    public String name() {
        return "rotate";
    }

    @Override
    public void execute() {
        WeightsManager weightsManager = fabric.createWeightsManager();
        ModelPredictor predictor = fabric.createPredictor();

        double[][] imageData = ImageHandler.load(new File(requiredArgs.get("data")));
        double[][] flattenImageData = new double[][]{ImageHandler.flattenImage(imageData)};
        double[][] rotatedImageData = AffineTransformations.rotateComplex(flattenImageData, rotationAngle);

        MatrixUtils.printMatrix(flattenImageData);
        System.out.println();
        MatrixUtils.printMatrix(rotatedImageData);

        System.out.println();
        System.out.println("Before weight rotation");
        System.out.println();
        System.out.println("Orig image");
        System.out.println(predictor.predict(flattenImageData[0]));
        System.out.println();
        System.out.println("Rotated image");
        System.out.println(predictor.predict(rotatedImageData[0]));

        int layerIndex = 0;
        double[][] origLayerWeights = weightsManager.getLayerWeights(layerIndex);
        double[][] rotatedWeights;
        rotatedWeights = AffineTransformations.rotateComplex(origLayerWeights, rotationAngle);
        weightsManager.setLayerWeights(layerIndex, rotatedWeights);

        System.out.println("After weight rotation on " + rotationAngle);
        System.out.println();
        System.out.println("Orig image");
        System.out.println(predictor.predict(flattenImageData[0]));
        System.out.println();
        System.out.println("Rotated image");
        System.out.println(predictor.predict(rotatedImageData[0])); 
    }

}
