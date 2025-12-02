package com.shavarushka.cli.commands;

import java.util.Map;

import com.shavarushka.affine.AffineTransformation;
import com.shavarushka.affine.MatrixUtils;
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

        double[] data = dataExtractor.extract();
        double[][] origLayerWeights = weightsManager.getLayerWeights(0);

        System.out.println();
        System.out.println("=".repeat(30) + " Before weight and data transformation " + "=".repeat(30));
        System.out.println();
        System.out.println("-".repeat(30) + " First hiden layer weights " + "-".repeat(30));
        System.out.println(getFirstLinesFrom(7, origLayerWeights));
        System.out.println();
        System.out.println("-".repeat(30) + " Neuron activations " + "-".repeat(30));
        MatrixUtils.printMatrix(neuronActivationHandler.getAllLayerActivationsAsArrays(data));
        System.out.println();
        System.out.println("-".repeat(30) + "Prediction:" + "-".repeat(30));
        System.out.println(predictor.predict(data));
        System.out.println("\n");

        ((ShearAffineTransformation) affineTransformation).setMatrixType(true);
        double[] rotatedData = affineTransformation.transform(new double[][]{data})[0];
        ((ShearAffineTransformation) affineTransformation).setMatrixType(false);
        double[][] rotatedWeights = affineTransformation.transform(origLayerWeights);

        weightsManager.setLayerWeights(0, rotatedWeights);

        System.out.println("=".repeat(30) + " After weight and data transformation " + "=".repeat(30));
        System.out.println();
        System.out.println("-".repeat(30) + " First hiden layer weights " + "-".repeat(30));
        System.out.println(getFirstLinesFrom(7, rotatedWeights));
        System.out.println();
        System.out.println("-".repeat(30) + "Neuron activations:" + "-".repeat(30));
        MatrixUtils.printMatrix(neuronActivationHandler.getAllLayerActivationsAsArrays(rotatedData));
        System.out.println();
        System.out.println("-".repeat(30) + "Prediction:" + "-".repeat(30));
        System.out.println(predictor.predict(rotatedData));

        weightsManager.setLayerWeights(0, origLayerWeights);

        if (Models.get(requiredArgs.get("model")).isVisualizable())
            visualization.execute();
    }

    private String getFirstLinesFrom(int count, double[][] matrix) {
        if (matrix == null || matrix.length == 0 || matrix[0] == null) {
            return "";
        }
        
        StringBuilder sb = new StringBuilder();
        
        int rowsToShow = Math.min(count, matrix.length);
        for (int i = 0; i < rowsToShow; i++) {
            int colsToShow = Math.min(count, matrix[i].length);
            for (int j = 0; j < colsToShow; j++) {
                sb.append(matrix[i][j]);
                if (j < colsToShow - 1) {
                    sb.append(" ");
                }
            }
            
            if (matrix[i].length > colsToShow) {
                sb.append(" ...");
            }
            
            if (i < rowsToShow - 1) {
                sb.append("\n");
            }
        }
        
        if (matrix.length > rowsToShow) {
            sb.append("\n...");
        }
        
        return sb.toString();
    }
}
