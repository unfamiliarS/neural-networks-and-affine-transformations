package com.shavarushka.cli;

import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;

public interface CLIOptions {
    
    static Options OPTIONS = createOptions();

    private static Options createOptions() {
        Options options = new Options();

        options.addOption(Option.builder()
                .longOpt("model")
                .required(true)
                .hasArg(true)
                .argName("model")
                .valueSeparator('=')
                .desc("The neuro model you want to work with")
                .build());

        options.addOption(Option.builder()
                .longOpt("mtype")
                .required(true)
                .hasArg(true)
                .argName("mtype")
                .valueSeparator('=')
                .desc("Type of neural network model you want to load")
                .build());

        options.addOption(Option.builder()
                .longOpt("data")
                .required(true)
                .hasArg(true)
                .argName("data")
                .valueSeparator('=')
                .desc("Sample data you want to process with the neural network")
                .build());

        options.addOption(Option.builder()
                .longOpt("rotate")
                .hasArg(true)
                .argName("angle")
                .valueSeparator('=')
                .desc("Affine rotation transformation example data ('x,y' point and MNIST 28x28 image supported) and neural network weights")
                .build());

        return options;
    }
}
