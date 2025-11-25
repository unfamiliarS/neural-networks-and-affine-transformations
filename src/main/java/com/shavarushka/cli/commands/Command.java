package com.shavarushka.cli.commands;

public interface Command {
    String name();
    void execute();
}
