package com.shavarushka.cli.commands;

import org.apache.commons.cli.HelpFormatter;

import com.shavarushka.cli.CLIOptions;

public class HelpCommand implements Command {

    @Override
    public String name() {
        return "help";
    }

    @Override
    public void execute() {
        new HelpFormatter().printHelp("affine-network [model-and-data] [transformation-type]", CLIOptions.OPTIONS);
    }
}
