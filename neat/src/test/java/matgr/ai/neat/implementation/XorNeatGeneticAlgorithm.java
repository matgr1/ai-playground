package matgr.ai.neat.implementation;

import matgr.ai.genetic.*;
import matgr.ai.math.clustering.Cluster;
import matgr.ai.neat.NeatGeneticAlgorithm;
import matgr.ai.neat.NeatGenome;
import matgr.ai.neat.crossover.NeatCrossoverSettings;
import matgr.ai.neat.mutation.NeatMutationSettings;
import matgr.ai.neat.speciation.SpeciationStrategy;
import org.apache.commons.math3.random.RandomGenerator;

import java.util.List;

public class XorNeatGeneticAlgorithm
        extends NeatGeneticAlgorithm<XorPopulation, XorSpecies, XorSpeciesMember, NeatGenome> {

    public XorNeatGeneticAlgorithm(RandomGenerator random,
                                   NeatCrossoverSettings crossoverSettings,
                                   NeatMutationSettings mutationSettings,
                                   SpeciationStrategy speciationStrategy) {
        super(random, crossoverSettings, mutationSettings, speciationStrategy);
    }

    @Override
    protected NeatGenome createNewGenome(int inputCount, int outputCount, double activationResponse) {
        return new NeatGenome(inputCount, outputCount, activationResponse);
    }

    @Override
    protected XorPopulation createPopulation(EvolutionContext context, List<XorSpecies> species, long generation) {
        return new XorPopulation(species, generation);
    }

    @Override
    protected XorSpecies createSpecies(Cluster<XorSpeciesMember> speciesMembers) {
        return new XorSpecies(speciesMembers);
    }

    @Override
    protected XorSpeciesMember createSpeciesMember(XorSpeciesMember template, NeatGenome genome) {
        return new XorSpeciesMember(genome);
    }
}
