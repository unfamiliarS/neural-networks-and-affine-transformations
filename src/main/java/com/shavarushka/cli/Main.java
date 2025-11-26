package com.shavarushka.cli;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.ParseException;

public class Main {
    public static void main(String[] args) {
        try {

            // String[] argss = new String[]{"--rotate=256", "--model=src/main/resources/simple-mnist.zip", "--mtype=mnist", "--data=src/main/resources/mnist-nums/2_000587.png"};
            String[] argss = new String[]{"--rotate=256", "--model=src/main/resources/triangle-classifier.zip", "--mtype=triangle", "--data=2.646525,2.611743"};

            CommandManager commandManager = new CommandManager(parseEnteredArgs(argss));
            commandManager.process();

        } catch (ParseException e) {
            System.err.println(e);
        }
    }

    private static CommandLine parseEnteredArgs(String[] args) throws ParseException {
        return new DefaultParser().parse(CLIOptions.OPTIONS, args, true);
    }
}
