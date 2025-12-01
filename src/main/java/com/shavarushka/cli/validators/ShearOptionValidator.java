package com.shavarushka.cli.validators;

import org.apache.commons.cli.CommandLine;

import com.shavarushka.cli.validators.exceptions.OptionValidationException;

public class ShearOptionValidator extends BaseOptionValidator {

    public ShearOptionValidator(CommandLine args, OptionValidator next) {
        super(args, next);
    }

    @Override
    public void validate() throws OptionValidationException {
        if (args.hasOption("shear")) {
            String scaleFactorStr = args.getOptionValue("shear");

            try {
                Double.parseDouble(scaleFactorStr);
            } catch (NumberFormatException e) {
                throw new OptionValidationException("Invalid shear: " + scaleFactorStr +
                    ". Must be a numeric value");
            }
        }

        validateNext();
    }
}
