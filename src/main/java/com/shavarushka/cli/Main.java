package com.shavarushka.cli;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.ParseException;

public class Main {
    public static void main(String[] args) {
        try {

            CommandManager commandManager = new CommandManager(parseEnteredArgs(args));
            commandManager.process();

        } catch (ParseException e) {
            System.err.println(e);
        }
    }

    private static CommandLine parseEnteredArgs(String[] args) throws ParseException {
        return new DefaultParser().parse(CLIOptions.OPTIONS, args, true);
    }
}
