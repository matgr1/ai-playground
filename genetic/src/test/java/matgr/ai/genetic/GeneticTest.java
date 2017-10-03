package matgr.ai.genetic;

import junit.framework.Test;
import junit.framework.TestCase;
import junit.framework.TestSuite;
import matgr.ai.genetic.utility.TestGeneticAlgorithmUtility;
import matgr.ai.genetic.utility.TestPopulationUtility;
import matgr.ai.genetic.implementation.TestGeneticAlgorithm;
import matgr.ai.genetic.implementation.TestPopulation;
import matgr.ai.genetic.implementation.TestSpecies;
import matgr.ai.genetic.implementation.TestSpeciesMember;
import matgr.ai.genetic.selection.SelectionStrategy;
import matgr.ai.math.DiscreteDistribution;
import matgr.ai.math.MathFunctions;
import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;

import java.util.*;

/**
 * Unit test for simple App.
 */
public class GeneticTest extends TestCase {

    private static final RandomGenerator random = new MersenneTwister();

    /**
     * Create the test case
     *
     * @param testName name of the test case
     */
    public GeneticTest(String testName) {

        super(testName);
    }

    /**
     * @return the suite of tests being tested
     */
    public static Test suite() {
        return new TestSuite(GeneticTest.class);
    }

    // TODO: turn these into actual tests...

    public void test1() {

        int populationSize = 100;
        int genomeLength = 10;
        double target = 0.0;

        List<NumericGenome> genomes = TestPopulationUtility.createRandomGenomes(
                random,
                populationSize,
                genomeLength);


        List<TestSpeciesMember> members = new ArrayList<>();
        for (NumericGenome genome : genomes) {
            members.add(new TestSpeciesMember(genome, target));
        }

        TestGeneticAlgorithm algorithm = TestGeneticAlgorithmUtility.createAlgorithm(random, target);

        EvolutionParameters evolutionParameters = TestGeneticAlgorithmUtility.getEvolutionParameters();
        SelectionStrategy selectionStrategy = TestGeneticAlgorithmUtility.getSelectionStrategy();
        EvolutionContext evolutionContext = new EvolutionContext(evolutionParameters, selectionStrategy);

        TestPopulation population = algorithm.createNewPopulation(evolutionContext, members);
        PopulationFitnessSnapshot fitnessSnapshot = PopulationFitnessSnapshot.create(population);

        DiscreteDistribution<FitnessItem<UUID>> fitnessDistribution =
                selectionStrategy.getSelectionDistribution(fitnessSnapshot, null);

        HashMap<UUID, Integer> genomeSelectionCounts = new HashMap<>();
        SortedMap<Double, Integer> fitnessCounts = new TreeMap<>();

        List<NumericGenome> fitnessSortedGenomes = new ArrayList<>();

        for (TestSpecies s : population.species()) {
            for (TestSpeciesMember m : s.members()) {
                fitnessSortedGenomes.add(m.genome());
            }
        }

        fitnessSortedGenomes.sort(Comparator.comparingDouble(fitnessSnapshot::getFitness));

        boolean groupedSelectionSampling = TestGeneticAlgorithmUtility.groupedSelectionSampling;

        for (int i = 0; i < 100000; i++) {

            FitnessItem<UUID> selected = fitnessDistribution.sample(random, groupedSelectionSampling);
            Double fitness = fitnessSnapshot.getFitness(selected.item);

            genomeSelectionCounts.merge(selected.item, 1, (a, b) -> a + b);
            fitnessCounts.merge(fitness, 1, (a, b) -> a + b);

        }

        for (NumericGenome value : fitnessSortedGenomes) {

            int count = 0;

            if (genomeSelectionCounts.containsKey(value.genomeId())) {
                count = genomeSelectionCounts.get(value.genomeId());
            }

            System.out.println(String.format("%f - %d", fitnessSnapshot.getFitness(value), count));
        }

        for (Map.Entry<Double, Integer> pair : fitnessCounts.entrySet()) {
            System.out.println(String.format("%f - %d", pair.getKey(), pair.getValue()));
        }

    }

    public void test2() {

        int populationSize = 100;
        int genomeLength = 10;
        double target = 0.0;

        double minFitness = 10000.0;
        int maxIterations = 100;

        List<NumericGenome> initialGenomes = TestPopulationUtility.createRandomGenomes(
                random,
                populationSize,
                genomeLength);

        List<TestSpeciesMember> initialMembers = new ArrayList<>();
        for (NumericGenome genome : initialGenomes) {
            initialMembers.add(new TestSpeciesMember(genome, target));
        }

        TestGeneticAlgorithm algorithm = TestGeneticAlgorithmUtility.createAlgorithm(random, target);

        EvolutionParameters evolutionParameters = TestGeneticAlgorithmUtility.getEvolutionParameters();
        SelectionStrategy selectionStrategy = TestGeneticAlgorithmUtility.getSelectionStrategy();

        GeneticAlgorithmResult<TestSpeciesMember> result = algorithm.solve(
                initialMembers,
                evolutionParameters,
                selectionStrategy,
                maxIterations,
                minFitness);

        double value = TestSpeciesMember.computeValue(result.bestMatch.item.genome());
        boolean reachedTargetValue = MathFunctions.fuzzyCompare(value, target, minFitness);

        assertTrue("Did not reach target item", reachedTargetValue);
        assertTrue("Solution did not converge", result.success);

    }
}
