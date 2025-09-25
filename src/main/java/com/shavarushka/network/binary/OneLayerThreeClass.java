package com.shavarushka.network.binary;

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class OneLayerThreeClass extends BinaryClassifier {

    public OneLayerThreeClass() {
        super();
    }

    @Override
    protected void buildModel() {
        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .seed(modelSeed)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Adam(0.1))
                .l2(1e-4)
                .list()
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.XENT)
                        .nIn(2)
                        .nOut(2)
                        .activation(Activation.SIGMOID)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .build();

        model = new MultiLayerNetwork(config);
        model.init();
    }

    @Override
    public double[] predict(double x1, double x2) {
        INDArray input = Nd4j.create(new double[][]{{x1, x2}});
        INDArray output = model.output(input);
        
        double[] probabilities = new double[3];
        probabilities[0] = output.getDouble(0);
        probabilities[1] = output.getDouble(1);
        probabilities[2] = output.getDouble(2);
        
        return probabilities;
    }

    @Override
    public DataSet generateDataSet(int numSamples, long seed) {
        Nd4j.getRandom().setSeed(seed);
        INDArray features = Nd4j.randn(numSamples, 2);
        INDArray labels = Nd4j.create(numSamples, 3);
        
        for (int i = 0; i < numSamples; i++) {
            double x1 = features.getDouble(i, 0);
            double x2 = features.getDouble(i, 1);
            double sum = x1 + x2;
            
            // Используем диапазон вместо точного равенства
            if (sum < -0.2) {
                labels.putScalar(new int[]{i, 0}, 1.0); // Класс 0: x1+x2 < -0.1
            } else if (sum > 0.2) {
                labels.putScalar(new int[]{i, 1}, 1.0); // Класс 1: x1+x2 > 0.1
            } else {
                labels.putScalar(new int[]{i, 2}, 1.0); // Класс 2: -0.1 <= x1+x2 <= 0.1
            }
        }
        
        return new DataSet(features, labels);
    }

    public void saveToCSV(DataSet dataset, String filename) {
        try (PrintWriter writer = new PrintWriter(new FileWriter(filename))) {
            writer.println("x1,x2,label");
            
            INDArray features = dataset.getFeatures();
            INDArray labels = dataset.getLabels();
            
            for (int i = 0; i < features.rows(); i++) {
                double x1 = features.getDouble(i, 0);
                double x2 = features.getDouble(i, 1);
                
                int label = 0;
                if (labels.getDouble(i, 1) > 0.5)
                    label = 1;
                else if (labels.getDouble(i, 2) > 0.5)
                    label = 2;
                
                writer.printf("%.6f,%.6f,%d%n", x1, x2, label);
            }
            
            System.out.println("Датасет сохранен в: " + filename);
        } catch (IOException e) {
            System.err.println("Ошибка при сохранении датасета: " + e.getMessage());
        }
    }
}
