package com.shavarushka.network.api;

public interface Generator<T> {
    T generate(int numSamples);
}
