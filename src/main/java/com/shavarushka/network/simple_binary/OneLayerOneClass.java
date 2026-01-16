package com.shavarushka.network.simple_binary;

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

public class OneLayerOneClass extends BinaryClassifier {

    public OneLayerOneClass() {
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
                        .nOut(1)
                        .activation(Activation.SIGMOID)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .build();

        model = new MultiLayerNetwork(config);
        model.init();
    }

    public double[] predict(double x1, double x2) {
        INDArray input = Nd4j.create(new double[][]{{x1, x2}});
        INDArray output = model.output(input);
        return new double[]{output.getDouble(0)};
    }

    @Override
    public DataSet generateDataSet(int numSamples, long seed) {
        Nd4j.getRandom().setSeed(seed);
        INDArray features = Nd4j.randn(numSamples, 2);

        // class = 1 if x1 + x2 >= 0
        INDArray labels = Nd4j.create(numSamples, 1);
        for (int i = 0; i < numSamples; i++) {
            double x1 = features.getDouble(i, 0);
            double x2 = features.getDouble(i, 1);
            double label = (x1 + x2 >= 0) ? 1.0 : 0.0;
            labels.putScalar(i, 0, label);
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
                double label = labels.getDouble(i, 0);
                writer.printf("%.6f,%.6f,%.0f%n", x1, x2, label);
            }

            System.out.println("Датасет сохранен в: " + filename);
        } catch (IOException e) {
            System.err.println("Ошибка при сохранении датасета: " + e.getMessage());
        }
    }
}
