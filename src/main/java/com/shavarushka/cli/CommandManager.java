package com.shavarushka.cli;

import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.Option;

import com.shavarushka.cli.commands.Command;
import com.shavarushka.cli.commands.RotationCommand;

public class CommandManager {

    private Map<String, Command> commands = new HashMap<>();
    private Map<String, String> requiredArgs;
    private CommandLine parsedCommands;

    public CommandManager(CommandLine cmd) {
        parsedCommands = cmd;
        requiredArgs = new RequiredArgsParser().parse();

        register(new RotationCommand(requiredArgs, parsedCommands.getOptionValue("rotate")));
    }

    private void register(Command command) {
        commands.put(command.name(), command);
    }

    public void process() {
        for (Map.Entry<String, Command> entry : commands.entrySet())
            if (commandShouldExecute(entry))
                entry.getValue().execute();
    }

    private boolean commandShouldExecute(Entry<String,Command> entry) {
        return parsedCommands.hasOption(entry.getKey());
    }


    private class RequiredArgsParser {

        Map<String, String> parse() {
            Map<String, String> reqArgs = new HashMap<>();

            for (Option option : parsedCommands.getOptions())
                if (option.isRequired())
                    reqArgs.put(option.getArgName(), option.getValue());

            return reqArgs;
        }
    }

}
