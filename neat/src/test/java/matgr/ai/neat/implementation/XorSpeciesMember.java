package matgr.ai.neat.implementation;

import matgr.ai.genetic.SpeciesMember;
import matgr.ai.neat.NeatGenome;

import java.util.ArrayList;
import java.util.List;

public class XorSpeciesMember implements SpeciesMember<NeatGenome> {

    private NeatGenome genome;

    public XorSpeciesMember(NeatGenome genome) {
        this.genome = genome;
    }

    @Override
    public NeatGenome genome() {
        return genome;
    }

    @Override
    public double computeFitness() {

        double sumOfErrorSquares = 0;

        for (int i = 0; i < XorConstants.testSize; i++) {

            for (int j = 0; j < XorConstants.testSize; j++) {

                double expectedValue = (double) (i ^ j);

                List<Double> inputs = new ArrayList<>();

                inputs.add((double) i);
                inputs.add((double) j);

                List<Double> result = genome.neuralNet.activateSingle(
                        inputs,
                        XorConstants.bias,
                        XorConstants.maxStepsPerActivation,
                        XorConstants.resetStateBeforeActivation);

                double value = result.get(0);

                if (Double.isNaN(value)) {
                    throw new AssertionError("Received NaN activation result");
                }

                if (Double.isInfinite(value)) {

                    sumOfErrorSquares = Double.POSITIVE_INFINITY;
                    break;

                } else {

                    double error = (value - expectedValue);
                    double errorSquared = error * error;

                    sumOfErrorSquares += errorSquared;
                }
            }

            if (Double.isInfinite(sumOfErrorSquares)) {
                break;
            }
        }

        if (Double.isInfinite(sumOfErrorSquares)) {
            return Double.NEGATIVE_INFINITY;
        }

        return 1.0 - Math.sqrt(sumOfErrorSquares);
    }
}
