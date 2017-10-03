package matgr.ai.neuralnet.feedforward;

import matgr.ai.genetic.NumericGeneticAlgorithm;
import matgr.ai.genetic.NumericCrossoverSettings;
import matgr.ai.genetic.NumericMutationSettings;
import matgr.ai.genetic.Population;
import matgr.ai.genetic.Species;
import matgr.ai.genetic.SpeciesMember;
import org.apache.commons.math3.random.RandomGenerator;

public abstract class FeedForwardNeuralNetGeneticAlgorithm<
        PopulationT extends Population<SpeciesT>,
        SpeciesT extends Species<SpeciesMemberT>,
        SpeciesMemberT extends SpeciesMember<GenomeT>,
        GenomeT extends FeedForwardNeuralNetGenome>
        extends NumericGeneticAlgorithm<PopulationT, SpeciesT, SpeciesMemberT, GenomeT> {

    protected FeedForwardNeuralNetGeneticAlgorithm(RandomGenerator random,
                                                   NumericCrossoverSettings crossoverSettings,
                                                   NumericMutationSettings mutationSettings) {
        super(random, crossoverSettings, mutationSettings);
    }

}

