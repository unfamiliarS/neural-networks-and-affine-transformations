package com.shavarushka.cli.validators;

import java.util.List;

import org.apache.commons.cli.CommandLine;

import com.shavarushka.cli.validators.exceptions.OptionValidationException;
import com.shavarushka.network.api.Models;

public class ModelOptionValidator extends BaseOptionValidator {

    private static List<Models> SUPPORTED_MODELS = List.of(Models.values());;

    public ModelOptionValidator(CommandLine args, OptionValidator next) {
        super(args, next);
    }

    @Override
    public void validate() throws OptionValidationException {
        String model = args.getOptionValue("model");
        try {
            Models.get(model);
        } catch (IllegalArgumentException e) {
            throw new OptionValidationException(
                String.format("Unknown model type: '%s'. Supported types: %s",
                    model, SUPPORTED_MODELS));
        }

        validateNext();
    }
}
