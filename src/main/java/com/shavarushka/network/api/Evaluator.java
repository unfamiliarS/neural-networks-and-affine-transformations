package com.shavarushka.network.api;

import org.nd4j.evaluation.classification.Evaluation;

public interface Evaluator {
    Evaluation evaluate();
}
