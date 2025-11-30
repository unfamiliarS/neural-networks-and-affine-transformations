package com.shavarushka.cli.commands;

import java.util.Map;

import com.shavarushka.affine.AffineTransformation;
import com.shavarushka.affine.MatrixUtils;
import com.shavarushka.affine.ScaleAffineTransformation;
import com.shavarushka.network.api.ModelLoader;
import com.shavarushka.network.api.ModelPredictor;
import com.shavarushka.network.api.NeuronActivationHandler;
import com.shavarushka.network.api.WeightsManager;
import com.shavarushka.network.api.fabric.ModelFabric;
import com.shavarushka.network.api.fabric.ModelFactoryOfFactory;

public class ScaleCommand implements Command {

    private DataExtractionStrategy dataExtractor;
    private ModelFabric fabric;

    private double scaleFactor;

    private Map<String,String> requiredArgs;

    public ScaleCommand(Map<String,String> requiredArgs, String scaleFactor) {
        this.scaleFactor = scaleFactor != null ? Double.parseDouble(scaleFactor) : 0;
        this.requiredArgs = requiredArgs;
    }

    @Override
    public String name() {
        return "scale";
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

        AffineTransformation affineTransformation = new ScaleAffineTransformation()
                                                        .scaleFactor(scaleFactor);

        double[][] data = new double[][]{dataExtractor.extract()};
        ((ScaleAffineTransformation) affineTransformation).setMatrixType(true);
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
        ((ScaleAffineTransformation) affineTransformation).setMatrixType(false);
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
    }
}
