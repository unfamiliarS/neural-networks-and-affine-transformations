package com.shavarushka.cli.validators;

import java.util.regex.Pattern;

import org.apache.commons.cli.CommandLine;

import com.shavarushka.cli.validators.exceptions.OptionValidationException;

public class ImageDataOptionValidator extends BaseOptionValidator {

    private FileValidator fileValidator;

    public ImageDataOptionValidator(CommandLine args, OptionValidator next, FileValidator fileValidator) {
        super(args, next);
        this.fileValidator = fileValidator;
    }

    @Override
    public void validate() throws OptionValidationException {
        if (dataIsImage()) {
            fileValidator.validate();
        }

        validateNext();
    }

    private boolean dataIsImage() {
        String data = args.getOptionValue("data");
        Pattern pathPattern = Pattern.compile(".*[/\\\\].*");
        return pathPattern.matcher(data).matches();
    }
}
