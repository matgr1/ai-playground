package matgr.ai.neat.implementation;

import matgr.ai.genetic.crossover.CrossoverType;
import matgr.ai.genetic.mutation.MutationType;

public class XorConstants {

    public static final int testSize = 2;
    public static final double successFitnessThreshold = 0.9;

    public static final double bias = -1;

    public static final int maxStepsPerActivation = 10;
    public static final boolean resetStateBeforeActivation = true;

    public static final int populationSize = 500;
    public static final int generations = 1000;

    public static final int inputCount = 2;
    public static final int outputCount = 1;

    public static final double eliteProportion = 0.05;
    public static final int eliteCopies = 1;
    public static final double asexualReproductionProportion = 0.1;
    public static final double sexualReproductionProportion = 0.75;
    public static final double interSpeciesSexualReproductionProportion = 0.1;

    public static final double selectivePressure = 0.5;

    public static final double weightRange = 5.0;

    public static final double crossoverRate = 0.7;
    public static final double connectionCrossoverDisableRate = 0.75;
    public static final CrossoverType crossoverType = CrossoverType.WholeArithmetic;// Heuristic;

    public static final double mutationRate = 0.1;
    public static final MutationType mutationType = MutationType.Uniform;// NonUniform;
    public static final double nonUniformMutationAlpha = 10.0;
    public static final double nonUniformMutationGenerationFactor = 0.1;
    public static final double maxMutation = 0.25;

    public static final double neatAddNodeProbability = 9.0;
    public static final double neatRemoveNodeProbability = 1.0;
    public static final double neatAddConnectionProbability = 18.0;
    public static final double neatRemoveConnectionProbability = 9.0;
    public static final double neatMutateWeightProbability = 70.0;

    public static final int initialClusterCount = 10;
    public static final double speciationExcessFactor = 1.0;
    public static final double speciationDisjointFactor = 1.0;
    public static final double speciationWeightFactor = 0.4;
}
