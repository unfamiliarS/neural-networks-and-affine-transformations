package com.shavarushka.cli.validators;

import java.io.File;

import com.shavarushka.cli.validators.exceptions.OptionValidationException;

public class FileValidator implements OptionValidator {

    private String filePath;

    public FileValidator(String file) {
        filePath = file;
    }

    public void validate() throws OptionValidationException {
        validateFileExists();
        validateFileReadable();
    }

    public void validateFileExists() throws OptionValidationException {
        File file = new File(filePath);
        if (!file.exists()) {
            throw new OptionValidationException("File does not exist: " + filePath);
        }
    }

    public void validateFileReadable() throws OptionValidationException {
        File file = new File(filePath);
        if (!file.canRead()) {
            throw new OptionValidationException("Cannot read file: " + filePath);
        }
    }
}

