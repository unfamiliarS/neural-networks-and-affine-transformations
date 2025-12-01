package com.shavarushka.cli.commands;

import java.util.Map;

import com.shavarushka.affine.AffineTransformation;
import com.shavarushka.affine.MatrixUtils;
import com.shavarushka.affine.ScaleAffineTransformation;
import com.shavarushka.affine.ShearAffineTransformation;
import com.shavarushka.network.api.ModelPredictor;
import com.shavarushka.network.api.Models;
import com.shavarushka.network.api.NeuronActivationHandler;
import com.shavarushka.network.api.WeightsManager;
import com.shavarushka.network.api.fabric.ModelFabric;
import com.shavarushka.network.api.fabric.ModelFactoryOfFactory;

public class ShearCommand implements Command {

    private DataExtractionStrategy dataExtractor;
    private ModelFabric fabric;

    private double shear;

    private Map<String,String> requiredArgs;

    private Command visualization;

    public ShearCommand(Map<String,String> requiredArgs, String shear) {
        this.shear = shear != null ? Double.parseDouble(shear) : 0;
        this.requiredArgs = requiredArgs;
        visualization = new VisualizationCommand(requiredArgs, "shear", shear);
    }

    @Override
    public String name() {
        return "shear";
    }

    private void lazyInit() {
        dataExtractor = requiredArgs.get("model").endsWith("mnist") ?
                            new ImageDataExtraction(requiredArgs.get("data")) :
                            new PointDataExtraction(requiredArgs.get("data"));

        fabric = ModelFactoryOfFactory.createFabric(requiredArgs.get("model"));
    }

    @Override
    public void execute() {
        lazyInit();

        WeightsManager weightsManager = fabric.createWeightsManager();
        NeuronActivationHandler neuronActivationHandler = fabric.createNeuronActivationHander();
        ModelPredictor predictor = fabric.createPredictor();

        AffineTransformation affineTransformation = new ShearAffineTransformation()
                                                        .shear(shear);

        double[][] data = new double[][]{dataExtractor.extract()};
        ((ShearAffineTransformation) affineTransformation).setMatrixType(true);
        double[][] rotatedData = affineTransformation.transform(data);

        System.out.println();
        MatrixUtils.printMatrix(data);
        System.out.println();
        MatrixUtils.printMatrix(rotatedData);

        System.out.println();
        System.out.println("Before weight transformation");
        System.out.println();
        System.out.println("Original data");
        System.out.println("Neuron activations:");
        MatrixUtils.printMatrix(neuronActivationHandler.getAllLayerActivationsAsArrays(data[0]));
        System.out.println();
        System.out.println(predictor.predict(data[0]));
        System.out.println();
        System.out.println("Transformed data");
        System.out.println("Neuron activations:");
        MatrixUtils.printMatrix(neuronActivationHandler.getAllLayerActivationsAsArrays(rotatedData[0]));
        System.out.println();
        System.out.println(predictor.predict(rotatedData[0]));

        int layerIndex = 0;
        double[][] origLayerWeights = weightsManager.getLayerWeights(layerIndex);
        ((ShearAffineTransformation) affineTransformation).setMatrixType(false);
        double[][] rotatedWeights = affineTransformation.transform(origLayerWeights);
        weightsManager.setLayerWeights(layerIndex, rotatedWeights);

        System.out.println();
        System.out.println("After weight transformation");
        System.out.println();
        System.out.println("Original data");
        System.out.println("Neuron activations:");
        MatrixUtils.printMatrix(neuronActivationHandler.getAllLayerActivationsAsArrays(data[0]));
        System.out.println();
        System.out.println(predictor.predict(data[0]));
        System.out.println();
        System.out.println("Transformed data");
        System.out.println("Neuron activations:");
        MatrixUtils.printMatrix(neuronActivationHandler.getAllLayerActivationsAsArrays(rotatedData[0]));
        System.out.println();
        System.out.println(predictor.predict(rotatedData[0]));

        weightsManager.setLayerWeights(layerIndex, origLayerWeights);

        if (Models.get(requiredArgs.get("model")).isVisualizable())
            visualization.execute();
    }
}
