package matgr.ai.genetic.crossover;

import org.apache.commons.math3.random.RandomGenerator;

public class CrossoverFunctions {

    public static double crossover(RandomGenerator random,
                                   CrossoverSettings settings,
                                   double fittestParentGene,
                                   double otherParentGene) {
        double result;

        switch (settings.getNumericCrossoverType()) {

            case WholeArithmetic: {
                double weight = random.nextDouble();
                result = (weight * fittestParentGene) + ((1.0 - weight) * otherParentGene);
            }
            break;

            case Heuristic: {
                double weight = random.nextDouble();
                result = (weight * (otherParentGene - fittestParentGene)) + fittestParentGene;
            }
            break;

            default: {
                throw new IllegalArgumentException("Invalid CrossoverType");
            }

        }

        return settings.clampValueToRange(result);
    }

}
