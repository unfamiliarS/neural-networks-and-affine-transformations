package com.shavarushka.cli.validators;

import java.util.Set;

import org.apache.commons.cli.CommandLine;

import com.shavarushka.cli.validators.exceptions.OptionValidationException;

public class ModelTypeOptionValidator extends BaseOptionValidator {

    private static Set<String> SUPPORTED_MODELS = Set.of(
        "mnist", "triangle", "multipletriangle"
    );

    public ModelTypeOptionValidator(CommandLine args, OptionValidator next) {
        super(args, next);
    }

    @Override
    public void validate() throws OptionValidationException {
        String model = args.getOptionValue("mtype");
        if (!SUPPORTED_MODELS.contains(model.toLowerCase())) {
            throw new OptionValidationException(
                String.format("Unknown model type: '%s'. Supported types: %s",
                    model, SUPPORTED_MODELS));
        }

        validateNext();
    }
}
