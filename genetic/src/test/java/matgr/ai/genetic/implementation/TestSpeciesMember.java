package matgr.ai.genetic.implementation;

import matgr.ai.genetic.NumericGenome;
import matgr.ai.genetic.SpeciesMember;

public class TestSpeciesMember implements SpeciesMember<NumericGenome> {

    public final double target;
    private final NumericGenome genome;

    public TestSpeciesMember(NumericGenome genome, double target) {
        this.genome = genome;
        this.target = target;
    }

    @Override
    public NumericGenome genome() {
        return genome;
    }

    @Override
    public double computeFitness() {

        double value = computeValue(genome);

        if (0.0 == value) {
            return Double.POSITIVE_INFINITY;
        }

        return 1.0 / Math.abs(target - value);

    }

    public static double computeValue(NumericGenome genome) {

        double value = 0.0;

        for( double gene : genome)
        {
            value += gene;
        }

        return value;

    }
}
