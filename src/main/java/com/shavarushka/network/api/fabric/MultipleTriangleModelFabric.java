package com.shavarushka.network.api.fabric;

import java.util.Arrays;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

import com.shavarushka.network.api.DataGenerator;
import com.shavarushka.network.api.GeneratedDataEvaluator;
import com.shavarushka.network.api.GeneratedDataTrainer;
import com.shavarushka.network.api.ModelEvaluator;
import com.shavarushka.network.api.ModelPredictor;
import com.shavarushka.network.api.ModelTrainer;
import com.shavarushka.network.multipletriangle.MultipleTriangleDataGenerator;
import com.shavarushka.network.multipletriangle.MultipleTriangleNetwork;
import com.shavarushka.network.multipletriangle.MultipleTrianglePredictor;

public class MultipleTriangleModelFabric extends ModelFabric {

    private static DataGenerator testDataGenerator = new MultipleTriangleDataGenerator(1000, 2, false);
    // static {
    //     MultipleTriangleDataGenerator.saveToCSV(testDataGenerator.generate(), "multiple-triangle-dataset.csv", 2);
    //     for (double[][] triangle : ((MultipleTriangleDataGenerator) testDataGenerator).getTrianglesVertices()) {
    //         System.out.println("triangle: ");
    //         for (double[] point : triangle) {
    //             System.out.println(Arrays.toString(point));
    //         }
    //         System.out.println();
    //     }
    // }
    private ModelEvaluator evaluator = new GeneratedDataEvaluator(network, testDataGenerator);

    public MultipleTriangleModelFabric() {
        super(MultipleTriangleNetwork.create());
    }

    public MultipleTriangleModelFabric(MultiLayerNetwork net) {
        super(net);
    }

    @Override
    public MultiLayerNetwork createNetwork() {
        return network;
    }

    @Override
    public ModelTrainer createTrainer() {
        DataGenerator trainDataGenerator = new MultipleTriangleDataGenerator(1000, 2, true);
        return new GeneratedDataTrainer(network, trainDataGenerator, evaluator);
    }

    @Override
    public ModelEvaluator createEvaluator() {
        return evaluator;
    }

    @Override
    public ModelPredictor createPredictor() {
        return new MultipleTrianglePredictor(network);
    }
}
