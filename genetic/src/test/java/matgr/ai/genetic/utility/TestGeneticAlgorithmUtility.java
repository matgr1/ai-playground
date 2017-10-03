package matgr.ai.genetic.utility;

import matgr.ai.genetic.EvolutionParameters;
import matgr.ai.genetic.NumericCrossoverSettings;
import matgr.ai.genetic.NumericMutationSettings;
import matgr.ai.genetic.crossover.CrossoverType;
import matgr.ai.genetic.implementation.TestGeneticAlgorithm;
import matgr.ai.genetic.mutation.MutationType;
import matgr.ai.genetic.selection.RouletteWheelSelectionStrategy;
import matgr.ai.genetic.selection.SelectionStrategy;
import org.apache.commons.math3.random.RandomGenerator;

public class TestGeneticAlgorithmUtility {

    // TODO: these are all defaults and should be named as such (and overriding them should be allowed
    //       in the calls to create things

    public static final double crossoverRate = 0.7;
    public static final double mutationRate = 0.1;
    public static final MutationType mutationType = MutationType.NonUniform;
    public static final double nonUniformMutationAlpha = 10.0;
    public static final double nonUniformMutationGenerationFactor = 1.0;
    public static final double maxMutation = 0.3;
    public static final CrossoverType crossoverType = CrossoverType.WholeArithmetic;

    public static final double eliteProportion = 0.05;
    public static final int eliteCopies = 1;
    public static final double asexualReproductionProportion = 0.1;
    public static final double sexualReproductionProportion = 0.75;
    public static final double interSpeciesSexualReproductionProportion = 0.1;

    public static final double valueRange = 1.0;

    public static final boolean groupedSelectionSampling = false;

    public static TestGeneticAlgorithm createAlgorithm(RandomGenerator random, double target) {

        NumericCrossoverSettings crossoverSettings = new NumericCrossoverSettings() {
            @Override
            public CrossoverType getNumericCrossoverType() {
                return crossoverType;
            }

            @Override
            public double getValueRange() {
                return valueRange;
            }

            @Override
            public double getProbability() {
                return crossoverRate;
            }
        };

        NumericMutationSettings mutationSettings = new NumericMutationSettings() {
            @Override
            public double getProbability() {
                return mutationRate;
            }

            @Override
            public MutationType getNumericMutationType() {
                return mutationType;
            }

            @Override
            public double getNonUniformMutationAlpha() {
                return nonUniformMutationAlpha;
            }

            @Override
            public double getNonUniformMutationGenerationFactor() {
                return nonUniformMutationGenerationFactor;
            }

            @Override
            public double getMaxMutation() {
                return maxMutation;
            }

            @Override
            public double getValueRange() {
                return valueRange;
            }
        };

        return new TestGeneticAlgorithm(random, target, crossoverSettings, mutationSettings);
    }

    public static EvolutionParameters getEvolutionParameters() {

        return new EvolutionParameters() {
            @Override
            public double getEliteProportion() {
                return eliteProportion;
            }

            @Override
            public int getEliteCopies() {
                return eliteCopies;
            }

            @Override
            public double getAsexualReproductionProportion() {
                return asexualReproductionProportion;
            }

            @Override
            public double getSexualReproductionProportion() {
                return sexualReproductionProportion;
            }

            @Override
            public double getInterSpeciesSexualReproductionProportion() {
                return interSpeciesSexualReproductionProportion;
            }
        };
    }

    public static SelectionStrategy getSelectionStrategy() {
        return new RouletteWheelSelectionStrategy(groupedSelectionSampling);
    }
}
