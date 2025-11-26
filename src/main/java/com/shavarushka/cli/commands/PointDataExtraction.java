package com.shavarushka.cli.commands;

import java.util.InputMismatchException;

class PointDataExtraction implements DataExtractionStrategy {

    String pointCoordinates;

    public PointDataExtraction(String data) {
        pointCoordinates = data;
    }

    @Override
    public double[] extract() {
        String[] xy = pointCoordinates.strip().split(",");
        double[] point = new double[2];
        if (xy.length == 2) {
            try {
                point[0] = Double.parseDouble(xy[0]);
                point[1] = Double.parseDouble(xy[1]);
                return point;
            } catch (NumberFormatException e) {
                throw new InputMismatchException(
                    String.format("Invalid number format in point: '%s'. Expected format: 'x,y'", pointCoordinates)
                );
            }
        }

        throw new InputMismatchException(
            String.format("Invalid point format: '%s'. Expected format: 'x,y'", pointCoordinates)
        );
    }
}
