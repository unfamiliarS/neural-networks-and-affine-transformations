package com.shavarushka.cli.validators;

import org.apache.commons.cli.CommandLine;

import com.shavarushka.cli.validators.exceptions.OptionValidationException;

public abstract class BaseOptionValidator implements OptionValidator {

    protected OptionValidator next;
    protected CommandLine args;

    protected BaseOptionValidator(CommandLine args, OptionValidator next) {
        this.args = args;
        this.next = next;
    }

    protected void validateNext() throws OptionValidationException {
        if (next != null) {
            next.validate();
        }
    }
}
