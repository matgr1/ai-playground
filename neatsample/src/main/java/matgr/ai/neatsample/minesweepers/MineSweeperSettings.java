package matgr.ai.neatsample.minesweepers;

import matgr.ai.genetic.DefaultEvolutionParameters;
import matgr.ai.genetic.EvolutionParameters;
import matgr.ai.genetic.crossover.CrossoverSettings;
import matgr.ai.genetic.crossover.CrossoverType;
import matgr.ai.genetic.crossover.DefaultCrossoverSettings;
import matgr.ai.genetic.mutation.DefaultMutationSettings;
import matgr.ai.genetic.mutation.MutationSettings;
import matgr.ai.genetic.mutation.MutationType;
import matgr.ai.genetic.selection.LinearRankingSelectionStrategy;
import matgr.ai.genetic.selection.SelectionStrategy;
import matgr.ai.math.MathFunctions;
import matgr.ai.neat.crossover.DefaultNeatCrossoverSettings;
import matgr.ai.neat.crossover.NeatCrossoverSettings;
import matgr.ai.neat.mutation.DefaultNeatMutationSettings;
import matgr.ai.neat.mutation.NeatMutationSettings;
import matgr.ai.neat.mutation.NeatStructuralMutationType;
import matgr.ai.neat.speciation.KMedoidsSpeciationStrategy;
import matgr.ai.neat.speciation.SpeciationStrategy;
import matgr.ai.neatsample.Size;

import java.util.HashMap;
import java.util.Map;

// TODO: these should be split up better...
public class MineSweeperSettings {

    public final int mineSweeperOutputCount = 2;

    public final int hiddenLayers = 1;
    public final double visionConeDistance = 0.25;
    public final double visionConeAngle = MathFunctions.degreesToRadians(60);
    public final int visionConeDivisions = 1;
    public final int minesPerVisionConeDivision = 4;

    public final int iterationsPerGeneration = 1000;//4000;//1000;//4000;
    public final double starvationPeriodFraction = 5 / 100.0;

    public int getStarvationPeriod() {
        return (int) Math.round(iterationsPerGeneration * starvationPeriodFraction);
    }

    public final int populationCount = 200 * 3;
    public final int mineCount = 100 * 3;
    public final double explodeyMineCountFraction = 1.0;

    public final Size minefieldSize = new Size(1.0, 1.0);

    public double getMaxTurnRate() {
        return visionConeAngle / 4.0;// 2.0 * Math.PI / 4.0;
    }

    public final double minSpeedForwards = 0.002;
    public final double maxSpeedForwards = 0.005;
    public final double minSpeedReverse = 0.001;
    public final double maxSpeedReverse = 0.002;

    public final double digestionPeriodFraction = 1.0 / 100.0;

    public int getDigestionPeriod() {
        return (int) Math.round(iterationsPerGeneration * digestionPeriodFraction);
    }

    public final double crossoverRate = 0.7;
    public final CrossoverType crossoverType = CrossoverType.WholeArithmetic;// Heuristic;
    public final double connectionCrossoverDisableRate = 0.75;

    public final double mutationRate = 0.1;
    public final MutationType mutationType = MutationType.Uniform;// NonUniform;
    public final double nonUniformMutationAlpha = 10.0;
    public final double nonUniformMutationGenerationFactor = 0.1;
    public final double maxMutation = 0.25;

    public final double neatAddNodeProbability = 9.0;
    public final double neatRemoveNodeProbability = 1.0;
    public final double neatAddConnectionProbability = 18.0;
    public final double neatRemoveConnectionProbability = 9.0;
    public final double neatMutateWeightProbability = 70.0;

    public final double eliteProportion = 0.05;
    public final int eliteCopies = 1;
    public final double asexualReproductionProportion = 0.1;
    public final double sexualReproductionProportion = 0.75;
    public final double interSpeciesSexualReproductionProportion = 0.1;

    public static final int initialClusterCount = 10;
    public final double speciationExcessFactor = 1.0;
    public final double speciationDisjointFactor = 1.0;
    public final double speciationWeightFactor = 0.4;
    public final double speciationDistanceThreshold = 3.0;
    public final boolean speciationDynamicDistanceThreshold = true;
    public final long speciationDynamicDistanceThresholdUpdateInterval = 1;
    public final int speciationDynamicDistanceThresholdMinSpeciesCount = 5;
    public final int speciationDynamicDistanceThresholdMaxSpeciesCount = 10;
    public final double speciationDynamicDistanceThresholdModifier = 0.1;
    public final double speciationDynamicDistanceThresholdMin = 0.5;
    public final int minGenomeNormalizationSize = 20;

    public EvolutionParameters getEvolutionParameters() {
        return new DefaultEvolutionParameters(
                eliteProportion,
                eliteCopies,
                asexualReproductionProportion,
                sexualReproductionProportion,
                interSpeciesSexualReproductionProportion);
    }

    public final SelectionStrategy selectionStrategy = new LinearRankingSelectionStrategy(0.5);
    //public readonly SelectionStrategy SelectionStrategy = new ExponentialRankingSelectionStrategy(0.5);

    public final int mineGestationPeriod = 30;

    public final double bias = -1;

    public final double activationResponse = 4.9;//1.0;
    public final double weightRange = 5.0;

    public final double mineSweeperRadius = 0.015 / 3.0;
    public final double mineRadiusFraction = 1.0 / 3.0;

    public double lineWidth() {
        return 0.0005;
    }

    public boolean highlightMinesInView = true;

    public int getExplodeyMineCount() {
        return (int) (mineCount * explodeyMineCountFraction);
    }

    public double getMineRadius() {
        return mineSweeperRadius * mineRadiusFraction;
    }

    public int getNeuronsPerHiddenLayer() {
        // TODO: try other methods...

        int inputCount = getMineSweeperInputCount();

        int neurons = (inputCount * 2 / 3) + mineSweeperOutputCount;
        return MathFunctions.clamp(neurons, 1, inputCount / 2);
    }

    public int getMineSweeperInputCount() {
        // TODO: separate number of explodey mines to check?

        //// NOTE: 2 for the sweeper direction, 2 for the sweeper position, 3 for each closest mine (2 for the mine position,
        ////		 1 for the score), 3 for each closest explodey mine (2 for the mine position, 1 for the score)
        //return 2 + 2 + (3 * CheckClosestNMines) + (3 * CheckClosestNMines);

        //// NOTE: 2 for the sweeper direction vector, 3 for each closest mine (2 for the mine distance vector,
        ////		 1 for the score)
        //return 2 + (3 * CheckClosestNMines);

        //return 2 + (3 * CheckClosestNMines);

        //// NOTE: this is for RNN with input sets (is this a valid method?)
        //return 2 + 3;

        // 2 for the sweeper position, 2 for the sweeper direction, 2 for each mine per vision cone direction
        return 2 + 2 + (2 * minesPerVisionConeDivision * visionConeDivisions);
    }

    public NeatCrossoverSettings getNeatCrossoverSettings() {

        return new DefaultNeatCrossoverSettings(
                crossoverRate,
                connectionCrossoverDisableRate,
                getNumericCrossoverSettings());
    }

    public NeatMutationSettings getNeatMutationSettings() {

        return new DefaultNeatMutationSettings(
                mutationRate,
                getNumericMutationSettings(),
                getMutationProbabilityProportions());
    }

    public CrossoverSettings getNumericCrossoverSettings() {

        return new DefaultCrossoverSettings(weightRange, crossoverType);
    }

    public MutationSettings getNumericMutationSettings() {

        return new DefaultMutationSettings(
                weightRange,
                mutationType,
                nonUniformMutationAlpha,
                nonUniformMutationGenerationFactor,
                maxMutation);
    }

    public Map<NeatStructuralMutationType, Double> getMutationProbabilityProportions() {

        Map<NeatStructuralMutationType, Double> probabilities = new HashMap<>();

        probabilities.put(NeatStructuralMutationType.AddNode, neatAddNodeProbability);
        probabilities.put(NeatStructuralMutationType.RemoveNode, neatRemoveNodeProbability);
        probabilities.put(NeatStructuralMutationType.AddConnection, neatAddConnectionProbability);
        probabilities.put(NeatStructuralMutationType.RemoveConnection, neatRemoveConnectionProbability);
        probabilities.put(NeatStructuralMutationType.MutateWeight, neatMutateWeightProbability);

        return probabilities;
    }

    public SpeciationStrategy getSpeciationStrategy() {
        return new KMedoidsSpeciationStrategy(
                speciationExcessFactor,
                speciationDisjointFactor,
                speciationWeightFactor,
                initialClusterCount);
    }
}