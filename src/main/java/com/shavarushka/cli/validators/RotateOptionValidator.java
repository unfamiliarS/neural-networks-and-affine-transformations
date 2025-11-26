package com.shavarushka.cli.validators;

import org.apache.commons.cli.CommandLine;

import com.shavarushka.cli.validators.exceptions.OptionValidationException;

public class RotateOptionValidator extends BaseOptionValidator {

    public RotateOptionValidator(CommandLine args, OptionValidator next) {
        super(args, next);
    }

    @Override
    public void validate() throws OptionValidationException {
        if (args.hasOption("rotate")) {
            String rotateValue = args.getOptionValue("rotate");

            try {
                double angle = Double.parseDouble(rotateValue);
                if (angle < -360 || angle > 360)
                    throw new OptionValidationException("Rotation angle must be between -360 and 360 degrees");
            } catch (NumberFormatException e) {
                throw new OptionValidationException("Invalid rotation angle: " + rotateValue +
                    ". Must be a numeric value");
            }
        }

        validateNext();
    }
}
