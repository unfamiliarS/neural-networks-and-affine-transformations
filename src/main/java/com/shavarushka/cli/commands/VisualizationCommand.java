package com.shavarushka.cli.commands;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.Map;

import com.shavarushka.network.api.Models;
import com.shavarushka.network.api.WeightsManager;
import com.shavarushka.network.api.fabric.ModelFabric;
import com.shavarushka.network.api.fabric.ModelFactoryOfFactory;

class VisualizationCommand implements Command {

    private Map<String,String> requiredArgs;
    private ModelFabric fabric;

    private String transformationType;
    private Double angle;
    private Double scale;
    private Double shear;

    public VisualizationCommand(Map<String,String> requiredArgs, String transformationType, String transformationValue) {
        this.requiredArgs = requiredArgs;
        this.transformationType = transformationType;
        this.angle = transformationValue != null ? Double.parseDouble(transformationValue) : null;
        this.scale = transformationValue != null ? Double.parseDouble(transformationValue) : null;
        this.shear = transformationValue != null ? Double.parseDouble(transformationValue) : null;
    }

    @Override
    public String name() {
        return "visualize";
    }

    private void lazyInit() {
        fabric = ModelFactoryOfFactory.createFabric(requiredArgs.get("model"));
    }

    @Override
    public void execute() {
        lazyInit();

        WeightsManager weightsManager = fabric.createWeightsManager();
        double[][] layerWeights = weightsManager.getLayerWeights(0);
        double[] layerBiases = weightsManager.getLayerBiases(0);

        try {
            String weightsStr = convertWeightsToString(layerWeights);
            String biasesStr = convertBiasesToString(layerBiases);
            String datasetPath = Models.get(requiredArgs.get("model")).getDatasetPath();

            ProcessBuilder pb = buildPythonCommand(datasetPath, weightsStr, biasesStr);

            Process process = pb.start();

            BufferedReader errorReader = new BufferedReader(new InputStreamReader(process.getErrorStream()));

            StringBuilder errorOutput = new StringBuilder();

            String line;
            while ((line = errorReader.readLine()) != null)
                errorOutput.append(line).append("\n");

            int exitCode = process.waitFor();

            if (errorOutput.length() > 0) {
                System.err.println("Python script errors:");
                System.err.println(errorOutput.toString());
            }

            if (exitCode != 0)
                System.err.println("Python script exited with error code: " + exitCode);

            errorReader.close();

        } catch (Exception e) {
            System.err.println("Error executing visualization command: " + e.getMessage());
            e.printStackTrace();
        }
    }

    private String convertWeightsToString(double[][] weights) {
        StringBuilder sb = new StringBuilder();
        sb.append("[");
        for (int i = 0; i < weights.length; i++) {
            sb.append("[");
            for (int j = 0; j < weights[i].length; j++) {
                sb.append(String.format("%f", weights[i][j]));
                if (j < weights[i].length - 1) {
                    sb.append(",");
                }
            }
            sb.append("]");
            if (i < weights.length - 1) {
                sb.append(",");
            }
        }
        sb.append("]");
        return sb.toString();
    }

    private String convertBiasesToString(double[] biases) {
        StringBuilder sb = new StringBuilder();
        sb.append("[");
        for (int i = 0; i < biases.length; i++) {
            sb.append(String.format("%f", biases[i]));
            if (i < biases.length - 1) {
                sb.append(",");
            }
        }
        sb.append("]");
        return sb.toString();
    }

    private ProcessBuilder buildPythonCommand(String datasetPath, String weights, String biases) {
        String pythonScriptPath = System.getenv("HOME") + "/.local/affine/visualization.py";

        ProcessBuilder pb = new ProcessBuilder(
            "python", pythonScriptPath,
            "--dataset", datasetPath,
            "--weights", weights,
            "--biases", biases,
            "--affineTransformation", transformationType
        );

        switch (transformationType.toLowerCase()) {
            case "rotate":
                if (angle != null) {
                    pb.command().add("--angle");
                    pb.command().add(angle.toString());
                }
                break;
            case "scale":
                if (scale != null) {
                    pb.command().add("--scale");
                    pb.command().add(scale.toString());
                }
                break;
            case "shear":
                if (shear != null) {
                    pb.command().add("--shear");
                    pb.command().add(shear.toString());
                }
                break;
        }

        return pb;
    }
}
