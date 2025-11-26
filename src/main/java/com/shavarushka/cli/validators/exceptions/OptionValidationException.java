package com.shavarushka.cli.validators.exceptions;

public class OptionValidationException extends Exception {
    
    public OptionValidationException(String message) {
        super(message);
    }
    
    public OptionValidationException(String message, Throwable cause) {
        super(message, cause);
    }
}
