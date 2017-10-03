package matgr.ai.neat;

import junit.framework.Test;
import junit.framework.TestCase;
import junit.framework.TestSuite;
import matgr.ai.genetic.*;
import matgr.ai.genetic.crossover.DefaultCrossoverSettings;
import matgr.ai.genetic.mutation.DefaultMutationSettings;
import matgr.ai.genetic.selection.LinearRankingSelectionStrategy;
import matgr.ai.genetic.selection.SelectionStrategy;
import matgr.ai.neat.crossover.DefaultNeatCrossoverSettings;
import matgr.ai.neat.crossover.NeatCrossoverSettings;
import matgr.ai.neat.implementation.XorConstants;
import matgr.ai.neat.implementation.XorNeatGeneticAlgorithm;
import matgr.ai.neat.implementation.XorPopulation;
import matgr.ai.neat.implementation.XorSpeciesMember;
import matgr.ai.neat.mutation.DefaultNeatMutationSettings;
import matgr.ai.neat.mutation.NeatMutationSettings;
import matgr.ai.neat.mutation.NeatStructuralMutationType;
import matgr.ai.neat.speciation.KMedoidsSpeciationStrategy;
import matgr.ai.neat.speciation.SpeciationStrategy;
import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;

import java.util.HashMap;
import java.util.Map;
import java.util.UUID;


/**
 * Unit test for simple App.
 */
public class NeatTest extends TestCase {

    private final EvolutionParameters evolutionParameters;
    private final SelectionStrategy selectionStrategy;

    private final XorNeatGeneticAlgorithm geneticAlgorithm;

    /**
     * Create the test case
     *
     * @param testName name of the test case
     */
    public NeatTest(String testName) {

        super(testName);

        RandomGenerator random = new MersenneTwister();

        evolutionParameters = new DefaultEvolutionParameters(
                XorConstants.eliteProportion,
                XorConstants.eliteCopies,
                XorConstants.asexualReproductionProportion,
                XorConstants.sexualReproductionProportion,
                XorConstants.interSpeciesSexualReproductionProportion);

        selectionStrategy = new LinearRankingSelectionStrategy(XorConstants.selectivePressure);

        NeatCrossoverSettings crossoverSettings = new DefaultNeatCrossoverSettings(
                XorConstants.crossoverRate,
                XorConstants.connectionCrossoverDisableRate,
                new DefaultCrossoverSettings(XorConstants.weightRange, XorConstants.crossoverType));

        NeatMutationSettings mutationSettings = new DefaultNeatMutationSettings(
                XorConstants.mutationRate,
                new DefaultMutationSettings(
                        XorConstants.weightRange,
                        XorConstants.mutationType,
                        XorConstants.nonUniformMutationAlpha,
                        XorConstants.nonUniformMutationGenerationFactor,
                        XorConstants.maxMutation),
                getMutationProbabilityProportions());

        SpeciationStrategy speciationStrategy = new KMedoidsSpeciationStrategy(
                XorConstants.speciationExcessFactor,
                XorConstants.speciationDisjointFactor,
                XorConstants.speciationWeightFactor);

        geneticAlgorithm = new XorNeatGeneticAlgorithm(
                random,
                crossoverSettings,
                mutationSettings,
                speciationStrategy);
    }

    private static Map<NeatStructuralMutationType, Double> getMutationProbabilityProportions() {

        Map<NeatStructuralMutationType, Double> probabilities = new HashMap<>();

        probabilities.put(NeatStructuralMutationType.AddNode, XorConstants.neatAddNodeProbability);
        probabilities.put(NeatStructuralMutationType.RemoveNode, XorConstants.neatRemoveNodeProbability);
        probabilities.put(NeatStructuralMutationType.AddConnection, XorConstants.neatAddConnectionProbability);
        probabilities.put(NeatStructuralMutationType.RemoveConnection, XorConstants.neatRemoveConnectionProbability);
        probabilities.put(NeatStructuralMutationType.MutateWeight, XorConstants.neatMutateWeightProbability);

        return probabilities;
    }

    /**
     * @return the suite of tests being tested
     */
    public static Test suite() {
        return new TestSuite(NeatTest.class);
    }

    // NOTE: test is not ready yet
    private boolean skip = true;

    // TODO: turn these into actual tests...
    public void test1() {

        if (skip) {
            return;
        }

        EvolutionContext evolutionContext = geneticAlgorithm.createEvolutionContext(
                evolutionParameters,
                selectionStrategy);

        XorPopulation population = geneticAlgorithm.createRandomPopulation(
                evolutionContext,
                XorConstants.populationSize,
                XorConstants.inputCount,
                XorConstants.outputCount,
                XorConstants.activationResponse);

        long generation = 0;

        FitnessItem<XorSpeciesMember> bestMember = null;

        for (; generation < XorConstants.generations; generation++) {

            PopulationFitnessSnapshot fitnessSnapshot = PopulationFitnessSnapshot.create(population);
            FitnessItem<UUID> curBestMember = fitnessSnapshot.genomesDescendingFitness.get(0);

            if ((null == bestMember) || (curBestMember.fitness > bestMember.fitness)) {

                XorSpeciesMember member = population
                        .getGenomeSpecies(curBestMember.item)
                        .getMember(curBestMember.item);

                bestMember = new FitnessItem<>(member, curBestMember.fitness);
            }

            if (curBestMember.fitness >= XorConstants.successFitnessThreshold) {
                break;
            }

            population = geneticAlgorithm.evolve(evolutionContext, population);
        }

        double bestFitness = bestMember.item.computeFitness();

        assertTrue(bestFitness >= XorConstants.successFitnessThreshold);
    }
}
