package com.shavarushka.cli.validators;

import java.util.regex.Pattern;

import org.apache.commons.cli.CommandLine;

import com.shavarushka.cli.validators.exceptions.OptionValidationException;

public class PointDataOptionValidator extends BaseOptionValidator {

    public PointDataOptionValidator(CommandLine args, OptionValidator next) {
        super(args, next);
    }

    @Override
    public void validate() throws OptionValidationException {
        if (!dataIsPoint())
            throw new OptionValidationException("Invalid data value");

        validateNext();
    }

    private boolean dataIsPoint() {
        String data = args.getOptionValue("data");
        Pattern pointPattern = Pattern.compile("^-?\\d+\\.\\d+,-?\\d+\\.\\d+$");
        return pointPattern.matcher(data).matches();
    }
}
