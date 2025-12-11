package com.shavarushka.cli;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.ParseException;

import com.shavarushka.cli.validators.OptionValidator;
import com.shavarushka.cli.validators.exceptions.OptionValidationException;

public class Main {
    public static void main(String[] args) {
        try {

            String[] argss = new String[]{"--rotate=37", "--model=two-triangles", "--data=3.0,3.0"};

            CommandLine parsedArgs = parseEnteredArgs(argss);
            // CommandLine parsedArgs = parseEnteredArgs(args);

            validateArgs(parsedArgs);

            processCommands(parsedArgs);

        } catch (ParseException | OptionValidationException e) {
            System.err.println(e);
        }
    }

    private static CommandLine parseEnteredArgs(String[] args) throws ParseException {
        return new DefaultParser().parse(CLIOptions.OPTIONS, args, true);
    }

    private static void validateArgs(CommandLine parsedArgs) throws OptionValidationException {
        OptionValidator validator = OptionValidator.createDefaultValidationChain(parsedArgs);
        validator.validate();
    }

    private static void processCommands(CommandLine parsedArgs) {
        CommandManager commandManager = new CommandManager(parsedArgs);
        commandManager.process();
    }
}
