package matgr.ai.genetic.implementation;

import matgr.ai.genetic.EvolutionContext;
import matgr.ai.genetic.NumericGeneticAlgorithm;
import matgr.ai.genetic.NumericCrossoverSettings;
import matgr.ai.genetic.NumericMutationSettings;
import matgr.ai.genetic.NumericGenome;
import matgr.ai.math.clustering.Cluster;
import org.apache.commons.math3.random.RandomGenerator;

import java.util.List;
import java.util.UUID;

public class TestGeneticAlgorithm extends NumericGeneticAlgorithm<
        TestPopulation,
        TestSpecies,
        TestSpeciesMember,
        NumericGenome> {

    public final double target;

    public TestGeneticAlgorithm(RandomGenerator random,
                                double target,
                                NumericCrossoverSettings crossoverSettings,
                                NumericMutationSettings mutationSettings)

    {
        super(random, crossoverSettings, mutationSettings);
        this.target = target;
    }

    @Override
    protected TestPopulation createPopulation(EvolutionContext context, List<TestSpecies> species, long generation) {
        return new TestPopulation(species, generation);
    }

    @Override
    protected TestSpecies createSpecies(Cluster<TestSpeciesMember> speciesMembers) {
        return new TestSpecies(speciesMembers);
    }

    @Override
    protected TestSpeciesMember createSpeciesMember(TestSpeciesMember template, NumericGenome genome) {

        if (template == null) {
            throw new IllegalStateException("createSpeciesMember not supported without template");
        }

        return new TestSpeciesMember(genome, target);
    }

    @Override
    protected NumericGenome createNewGenome(List<Double> genes, UUID genomeId) {
        return new NumericGenome(genomeId, genes);
    }
}