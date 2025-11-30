package com.shavarushka.cli.validators;

import org.apache.commons.cli.CommandLine;

import com.shavarushka.cli.validators.exceptions.OptionValidationException;

public class ScaleOptionValidator extends BaseOptionValidator {

    public ScaleOptionValidator(CommandLine args, OptionValidator next) {
        super(args, next);
    }

    @Override
    public void validate() throws OptionValidationException {
        if (args.hasOption("scale")) {
            String scaleFactorStr = args.getOptionValue("scale");

            try {
                Double.parseDouble(scaleFactorStr);
            } catch (NumberFormatException e) {
                throw new OptionValidationException("Invalid scale factor: " + scaleFactorStr +
                    ". Must be a numeric value");
            }
        }

        validateNext();
    }
}
