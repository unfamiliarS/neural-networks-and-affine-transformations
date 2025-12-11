package com.shavarushka.cli;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.ParseException;

import com.shavarushka.cli.commands.HelpCommand;
import com.shavarushka.cli.validators.OptionValidator;
import com.shavarushka.cli.validators.exceptions.OptionValidationException;

public class Main {
    public static void main(String[] args) {
        try {

            CommandLine parsedArgs = parseEnteredArgs(args);

            validateArgs(parsedArgs);

            processCommands(parsedArgs);

        } catch (ParseException | OptionValidationException e) {
            System.err.println(e);
            new HelpCommand().execute();
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
