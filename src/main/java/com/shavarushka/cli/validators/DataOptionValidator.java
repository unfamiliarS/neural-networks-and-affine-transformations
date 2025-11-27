package com.shavarushka.cli.validators;

import java.util.regex.Pattern;

import org.apache.commons.cli.CommandLine;

import com.shavarushka.cli.validators.exceptions.OptionValidationException;

public class DataOptionValidator extends BaseOptionValidator {

    private FileValidator fileValidator;

    public DataOptionValidator(CommandLine args, OptionValidator next, FileValidator fileValidator) {
        super(args, next);
        this.fileValidator = fileValidator;
    }

    @Override
    public void validate() throws OptionValidationException {
        if (dataIsImage())
            fileValidator.validate();
        else if (!dataIsPoint())
            throw new OptionValidationException("Invalid data value");

        validateNext();
    }

    private boolean dataIsImage() {
        String data = args.getOptionValue("data");
        Pattern pathPattern = Pattern.compile(".*[/\\\\].*");
        return pathPattern.matcher(data).matches();
    }

    private boolean dataIsPoint() {
        String data = args.getOptionValue("data");
        Pattern pointPattern = Pattern.compile("^-?\\d+\\.\\d+,-?\\d+\\.\\d+$");
        return pointPattern.matcher(data).matches();
    }
}
