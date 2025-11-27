package com.shavarushka.cli.commands;

import java.util.Map;

import com.shavarushka.affine.AffineTransformations;
import com.shavarushka.affine.MatrixUtils;
import com.shavarushka.network.api.ModelLoader;
import com.shavarushka.network.api.ModelPredictor;
import com.shavarushka.network.api.NeuronActivationHandler;
import com.shavarushka.network.api.WeightsManager;
import com.shavarushka.network.api.fabric.ModelFabric;
import com.shavarushka.network.api.fabric.ModelFactoryOfFactory;

public class RotationCommand implements Command {

    private DataExtractionStrategy dataExtractor;
    private ModelFabric fabric;

    private double rotationAngle;

    public RotationCommand(Map<String,String> requiredArgs, String rotationAngle) {
        this.rotationAngle = Double.parseDouble(rotationAngle);

        dataExtractor = requiredArgs.get("mtype").equals("mnist") ?
                            new ImageDataExtraction(requiredArgs.get("data")) :
                            new PointDataExtraction(requiredArgs.get("data"));

        fabric = ModelFactoryOfFactory.createFabric(
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

        double[][] data = new double[][]{dataExtractor.extract()};
        double[][] rotatedData = AffineTransformations.rotateComplex(data, rotationAngle);

        System.out.println();
        MatrixUtils.printMatrix(data);
        System.out.println();
        MatrixUtils.printMatrix(rotatedData);

        System.out.println();
        System.out.println("Before weight rotation");
        System.out.println();
        System.out.println("Original data");
        System.out.println("Neuron activations:");
        MatrixUtils.printMatrix(NeuronActivationHandler.getAllLayerActivationsAsArrays(fabric.createNetwork(), data[0]));
        System.out.println();
        System.out.println(predictor.predict(data[0]));
        System.out.println();
        System.out.println("Rotated data");
        System.out.println("Neuron activations:");
        MatrixUtils.printMatrix(NeuronActivationHandler.getAllLayerActivationsAsArrays(fabric.createNetwork(), rotatedData[0]));
        System.out.println();
        System.out.println(predictor.predict(rotatedData[0]));

        int layerIndex = 0;
        double[][] origLayerWeights = weightsManager.getLayerWeights(layerIndex);
        double[][] rotatedWeights;
        rotatedWeights = AffineTransformations.rotateComplex(origLayerWeights, rotationAngle);
        weightsManager.setLayerWeights(layerIndex, rotatedWeights);

        System.out.println();
        System.out.println("After weight rotation on " + rotationAngle);
        System.out.println();
        System.out.println("Original data");
        System.out.println("Neuron activations:");
        MatrixUtils.printMatrix(NeuronActivationHandler.getAllLayerActivationsAsArrays(fabric.createNetwork(), data[0]));
        System.out.println();
        System.out.println(predictor.predict(data[0]));
        System.out.println();
        System.out.println("Rotated data");
        System.out.println("Neuron activations:");
        MatrixUtils.printMatrix(NeuronActivationHandler.getAllLayerActivationsAsArrays(fabric.createNetwork(), rotatedData[0]));
        System.out.println();
        System.out.println(predictor.predict(rotatedData[0]));
    }
}
