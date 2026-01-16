package com.shavarushka.cli.validators;

import org.apache.commons.cli.CommandLine;

import com.shavarushka.cli.validators.exceptions.OptionValidationException;

public interface OptionValidator {

    void validate() throws OptionValidationException;

    public static OptionValidator createDefaultValidationChain(CommandLine args) {
        var imageFileValidator = new FileValidator(args.getOptionValue("data"));
        var validator4 = new ScaleOptionValidator(args, null);
        var validator3 = new RotateOptionValidator(args, validator4);
        var validator2 = new DataOptionValidator(args, validator3, imageFileValidator);
        var validator1 = new ModelOptionValidator(args, validator2);
        return validator1;
    }
}
