package com.shavarushka.cli.validators;

import java.util.regex.Pattern;

import org.apache.commons.cli.CommandLine;

import com.shavarushka.cli.validators.exceptions.OptionValidationException;
import com.shavarushka.network.api.Models;
import static com.shavarushka.network.api.Models.*;

public class DataOptionValidator extends BaseOptionValidator {

    private FileValidator fileValidator;

    public DataOptionValidator(CommandLine args, OptionValidator next, FileValidator fileValidator) {
        super(args, next);
        this.fileValidator = fileValidator;
    }

    @Override
    public void validate() throws OptionValidationException {
        Models modelType = Models.get(args.getOptionValue("model"));
        String data = args.getOptionValue("data");

        if (isImageModel(modelType)) {
            if (!dataIsImage(data)) {
                throw new OptionValidationException(
                    String.format("Model '%s' expects image data, but got: %s", modelType, data)
                );
            }
            fileValidator.validate();
        }
        else if (isPointModel(modelType)) {
            if (!dataIsPoint(data)) {
                throw new OptionValidationException(
                    String.format("Model '%s' expects point data (format: x,y), but got: %s", modelType, data)
                );
            }
        }
        else {
            throw new OptionValidationException("Unsupported model type: " + modelType);
        }

        validateNext();
    }

    private boolean isImageModel(Models modelType) {
        return modelType.equals(MNIST) ||
            modelType.equals(SIMPLE_MNIST);
    }

    private boolean isPointModel(Models modelType) {
        return modelType.equals(TRIANGLE) ||
            modelType.equals(TWO_TRIANGLES);
    }

    private boolean dataIsImage(String data) {
        Pattern pathPattern = Pattern.compile(".*[/\\\\].*");
        return pathPattern.matcher(data).matches();
    }

    private boolean dataIsPoint(String data) {
        Pattern pointPattern = Pattern.compile("^-?\\d+\\.\\d+,-?\\d+\\.\\d+$");
        return pointPattern.matcher(data).matches();
    }
}
