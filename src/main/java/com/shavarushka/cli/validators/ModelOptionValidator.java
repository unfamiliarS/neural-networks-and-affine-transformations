package com.shavarushka.cli.validators;

import org.apache.commons.cli.CommandLine;

import com.shavarushka.cli.validators.exceptions.OptionValidationException;

public class ModelOptionValidator extends BaseOptionValidator {

    private FileValidator fileValidator;

    protected ModelOptionValidator(CommandLine args, OptionValidator next, FileValidator fileValidator) {
        super(args, next);
        this.fileValidator = fileValidator;
    }

    @Override
    public void validate() throws OptionValidationException {
        fileValidator.validate();

        validateNext();
    }
}
