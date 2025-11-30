package com.shavarushka.cli.commands;

import java.io.BufferedReader;
import java.util.Map;

public class VisualizationCommand implements Command {

    private Map<String, String> requiredArgs;
    private String transformationType;
    private Double angle;
    private Double scale;
    private Double shear;

    public VisualizationCommand(Map<String, String> requiredArgs, String transformationType, 
                              String angle, String scale, String shear) {
        this.requiredArgs = requiredArgs;
        this.transformationType = transformationType;
        this.angle = angle != null ? Double.parseDouble(angle) : null;
        this.scale = scale != null ? Double.parseDouble(scale) : null;
        this.shear = shear != null ? Double.parseDouble(shear) : null;
    }

    @Override
    public String name() {
        return "visualize";
    }

    @Override
    public void execute() {
        try {
            // Получаем веса и смещения из модели
            String weights = getWeightsFromModel();
            String biases = getBiasesFromModel();
            
            // Строим команду для Python скрипта
            ProcessBuilder pb = buildPythonCommand(weights, biases);
            
            // Запускаем процесс
            Process process = pb.start();
            
            // Читаем вывод
            BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
            String line;
            while ((line = reader.readLine()) != null) {
                System.out.println(line);
            }
            
            // Читаем ошибки
            BufferedReader errorReader = new BufferedReader(new InputStreamReader(process.getErrorStream()));
            while ((line = errorReader.readLine()) != null) {
                System.err.println(line);
            }
            
            int exitCode = process.waitFor();
            if (exitCode != 0) {
                System.err.println("Python script exited with error code: " + exitCode);
            }
            
        } catch (Exception e) {
            System.err.println("Error executing visualization command: " + e.getMessage());
            e.printStackTrace();
        }
    }

    private String getWeightsFromModel() {
        // Заглушка - в реальности нужно получить веса из модели
        // Для примера возвращаем тестовые веса
        return "[[-0.25,0.95],[0.27,0.14],[-0.01,-0.04],[1.41,-1.07],[0.05,-2.79],[-0.73,-0.77]]";
    }

    private String getBiasesFromModel() {
        // Заглушка - в реальности нужно получить смещения из модели
        return "[-2.46,-3.62,-2.81,0.02,2.08,5.07]";
    }

    private ProcessBuilder buildPythonCommand(String weights, String biases) {
        String pythonScriptPath = "/home/semyon/projects/neural-networks-and-affine-transformations/src/main/python/visualization.py";
        
        ProcessBuilder pb = new ProcessBuilder(
            "python", pythonScriptPath,
            "--weights", weights,
            "--biases", biases
        );
        
        // Добавляем параметры преобразования
        if (transformationType != null) {
            pb.command().add("--affineTransformation");
            pb.command().add(transformationType);
            
            switch (transformationType.toLowerCase()) {
                case "rotation":
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
        }
        
        return pb;
    }
}
