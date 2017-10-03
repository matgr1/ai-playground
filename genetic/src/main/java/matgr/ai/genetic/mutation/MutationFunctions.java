package matgr.ai.genetic.mutation;

import matgr.ai.math.RandomFunctions;
import org.apache.commons.math3.random.RandomGenerator;

public class MutationFunctions {

    public static double mutate(RandomGenerator random,
                                MutationSettings settings,
                                long currentGeneration,
                                double gene) {

        // TODO: more mutation methods (https://en.wikipedia.org/wiki/Mutation_(genetic_algorithm) and
        //		 http://www.mssanz.org.au/MODSIM97/Vol%204/Su.pdf)

        double mutationAmount;

        switch (settings.getNumericMutationType()) {

            case Uniform: {
                mutationAmount = RandomFunctions.nextDouble(
                        random,
                        -settings.getMaxMutation(),
                        settings.getMaxMutation());
            }
            break;

            case NonUniform: {
                // TODO: make this based on fitness (of the population or parents of the individual?) as well?

                double nonUniformFactor = settings.getNonUniformMutationGenerationFactor();
                double baseMutationAmount = 1.0 / ((nonUniformFactor * currentGeneration) + 1.0);

                // NOTE: mean = alpha / (alpha + beta)... see here:
                //         http://stats.stackexchange.com/questions/47771/what-is-the-intuition-behind-beta-distribution
                //		 and here:
                //         http://www.math.uah.edu/stat/apps/SpecialSimulation.html
                double alpha = settings.getNonUniformMutationAlpha();
                double beta = Math.max(0.5, (alpha - (baseMutationAmount * alpha)) / baseMutationAmount);

                double mutationFactor = RandomFunctions.sampleBetaDistribution(random, alpha, beta);

                double sign = RandomFunctions.nextSign(random);
                mutationAmount = sign * mutationFactor * settings.getMaxMutation();
            }
            break;

            default: {
                throw new IllegalArgumentException("Invalid MutationType");
            }

        }

        return settings.clampValueToRange(gene + mutationAmount);
    }

}
