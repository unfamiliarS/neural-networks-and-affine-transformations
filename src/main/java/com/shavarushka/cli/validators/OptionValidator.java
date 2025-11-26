package com.shavarushka.cli.validators;

import org.apache.commons.cli.CommandLine;

import com.shavarushka.cli.validators.exceptions.OptionValidationException;

public interface OptionValidator {
    void validate() throws OptionValidationException;

    public static OptionValidator createDefaultValidationChain(CommandLine args) {
        var imageFileValidator = new FileValidator(args.getOptionValue("data"));
        var modelFileValidator = new FileValidator(args.getOptionValue("model"));
        var validator5 = new RotateOptionValidator(args, null);
        var validator4 = new ImageDataOptionValidator(args, validator5, imageFileValidator);
        var validator3 = new PointDataOptionValidator(args, validator4);
        var validator2 = new ModelTypeOptionValidator(args, validator3);
        var validator1 = new ModelOptionValidator(args, validator2, modelFileValidator);
        return validator1;
    }
}
