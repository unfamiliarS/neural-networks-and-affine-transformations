package com.shavarushka.cli.commands;

import java.util.Map;

import com.shavarushka.affine.AffineTransformation;
import com.shavarushka.affine.MatrixUtils;
import com.shavarushka.affine.RotationAffineTransformation;
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

    private Map<String,String> requiredArgs;


    public RotationCommand(Map<String,String> requiredArgs, String rotationAngle) {
        this.rotationAngle = rotationAngle != null ? Double.parseDouble(rotationAngle) : 0;
        this.requiredArgs = requiredArgs;
    }

    @Override
    public String name() {
        return "rotate";
    }

    private void lazyInit() {
        dataExtractor = requiredArgs.get("mtype").equals("mnist") ?
                            new ImageDataExtraction(requiredArgs.get("data")) :
                            new PointDataExtraction(requiredArgs.get("data"));

        fabric = ModelFactoryOfFactory.createFabric(
            requiredArgs.get("mtype"),
            ModelLoader.load(requiredArgs.get("model"))
        );
    }

    @Override
    public void execute() {
        lazyInit();

        WeightsManager weightsManager = fabric.createWeightsManager();
        NeuronActivationHandler neuronActivationHandler = fabric.createNeuronActivationHander();
        ModelPredictor predictor = fabric.createPredictor();

        AffineTransformation affineTransformation = new RotationAffineTransformation()
                                                            .angle(rotationAngle);

        double[][] data = new double[][]{dataExtractor.extract()};
        double[][] rotatedData = affineTransformation.transform(data);

        System.out.println();
        MatrixUtils.printMatrix(data);
        System.out.println();
        MatrixUtils.printMatrix(rotatedData);

        System.out.println();
        System.out.println("Before weight rotation");
        System.out.println();
        System.out.println("Original data");
        System.out.println("Neuron activations:");
        MatrixUtils.printMatrix(neuronActivationHandler.getAllLayerActivationsAsArrays(data[0]));
        System.out.println();
        System.out.println(predictor.predict(data[0]));
        System.out.println();
        System.out.println("Rotated data");
        System.out.println("Neuron activations:");
        MatrixUtils.printMatrix(neuronActivationHandler.getAllLayerActivationsAsArrays(rotatedData[0]));
        System.out.println();
        System.out.println(predictor.predict(rotatedData[0]));

        int layerIndex = 0;
        double[][] origLayerWeights = weightsManager.getLayerWeights(layerIndex);
        double[][] rotatedWeights = affineTransformation.transform(origLayerWeights);
        weightsManager.setLayerWeights(layerIndex, rotatedWeights);

        System.out.println();
        System.out.println("After weight rotation on " + rotationAngle);
        System.out.println();
        System.out.println("Original data");
        System.out.println("Neuron activations:");
        MatrixUtils.printMatrix(neuronActivationHandler.getAllLayerActivationsAsArrays(data[0]));
        System.out.println();
        System.out.println(predictor.predict(data[0]));
        System.out.println();
        System.out.println("Rotated data");
        System.out.println("Neuron activations:");
        MatrixUtils.printMatrix(neuronActivationHandler.getAllLayerActivationsAsArrays(rotatedData[0]));
        System.out.println();
        System.out.println(predictor.predict(rotatedData[0]));
    }
}
