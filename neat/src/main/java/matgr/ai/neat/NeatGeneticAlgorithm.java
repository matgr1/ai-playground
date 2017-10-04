package matgr.ai.neat;

import matgr.ai.genetic.*;
import matgr.ai.genetic.selection.SelectionStrategy;
import matgr.ai.math.RandomFunctions;
import matgr.ai.neat.crossover.NeatCrossoverFunctions;
import matgr.ai.neat.crossover.NeatCrossoverSettings;
import matgr.ai.neat.mutation.NeatMutationFunctions;
import matgr.ai.neat.mutation.NeatMutationSettings;
import matgr.ai.neat.speciation.SpeciationStrategy;
import matgr.ai.neuralnet.cyclic.Neuron;
import matgr.ai.neuralnet.cyclic.NeuronType;
import org.apache.commons.math3.random.RandomGenerator;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

// TODO: try different activation functions

public abstract class NeatGeneticAlgorithm<
        PopulationT extends Population<SpeciesT>,
        SpeciesT extends Species<SpeciesMemberT>,
        SpeciesMemberT extends SpeciesMember<NeatGenomeT>,
        NeatGenomeT extends NeatGenome>
        extends GeneticAlgorithm<PopulationT, SpeciesT, SpeciesMemberT, NeatGenomeT> {

    public final NeatCrossoverSettings crossoverSettings;

    public final NeatMutationSettings mutationSettings;

    public final SpeciationStrategy speciationStrategy;

    protected NeatGeneticAlgorithm(RandomGenerator random,
                                   NeatCrossoverSettings crossoverSettings,
                                   NeatMutationSettings mutationSettings,
                                   SpeciationStrategy speciationStrategy) {

        super(random);

        if (null == crossoverSettings) {
            throw new IllegalArgumentException("crossoverSettings not provided");
        }
        if (null == mutationSettings) {
            throw new IllegalArgumentException("mutationSettings not provided");
        }
        if (null == speciationStrategy) {
            throw new IllegalArgumentException("speciationStrategy not provided");
        }

        this.crossoverSettings = crossoverSettings;
        this.mutationSettings = mutationSettings;
        this.speciationStrategy = speciationStrategy;
    }

    public PopulationT createRandomPopulation(EvolutionContext context,
                                              int populationSize,
                                              int inputCount,
                                              int outputCount) {

        return createRandomPopulation(
                getNeatEvolutionContext(context),
                populationSize,
                inputCount,
                outputCount);
    }

    private PopulationT createRandomPopulation(NeatEvolutionContext context,
                                               int populationSize,
                                               int inputCount,
                                               int outputCount) {

        List<SpeciesMemberT> genomes = new ArrayList<>();

        for (int i = 0; i < populationSize; i++) {

            NeatGenomeT genome = createRandomGenome(random, inputCount, outputCount);

            for (Neuron outputNode : genome.neuralNet.neurons.values(NeuronType.Output)) {

                addRandomConnection(context, genome, genome.neuralNet.biasNeuron(), outputNode);

                for (Neuron inputNode : genome.neuralNet.neurons.values(NeuronType.Input)) {

                    addRandomConnection(context, genome, inputNode, outputNode);
                }
            }

            SpeciesMemberT member = createSpeciesMember(null, genome);
            genomes.add(member);
        }

        return createNewPopulation(context, genomes);
    }

    private void addRandomConnection(NeatEvolutionContext context,
                                     NeatGenomeT genome,
                                     Neuron source,
                                     Neuron target) {

        NeatMutationFunctions.addConnection(
                genome,
                source.id,
                target.id,
                mutationSettings.getConnectionWeightsMutationSettings().getRandomValueInRange(random),
                context.innovationMap);
    }

    @Override
    public final NeatEvolutionContext createEvolutionContext(EvolutionParameters evolutionParameters,
                                                             SelectionStrategy selectionStrategy) {
        return new NeatEvolutionContext(evolutionParameters, selectionStrategy);
    }

    protected NeatEvolutionContext getNeatEvolutionContext(EvolutionContext context){

        // TODO: is there a better way of handling this?
        @SuppressWarnings("unchecked")
        NeatEvolutionContext neatContext = (NeatEvolutionContext) context;

        return neatContext;
    }

    @Override
    protected List<SpeciesT> speciate(EvolutionContext context,
                                      List<SpeciesMemberT> members,
                                      PopulationT previousPopulation) {
        return speciate( members, previousPopulation);
    }

    private List<SpeciesT> speciate(List<SpeciesMemberT> members,
                                    PopulationT previousPopulation) {
        return speciationStrategy.speciate(members, previousPopulation, this::createSpecies);
    }

    @Override
    protected List<SpeciesMemberT> createOffspringAsexual(EvolutionContext context,
                                                          FitnessItem<SpeciesMemberT> parent,
                                                          long currentGeneration,
                                                          int count) {
        return createOffspringAsexual(getNeatEvolutionContext(context), parent, currentGeneration, count);
    }

    private List<SpeciesMemberT> createOffspringAsexual(NeatEvolutionContext context,
                                                        FitnessItem<SpeciesMemberT> parent,
                                                        long currentGeneration,
                                                        int count) {

        List<SpeciesMemberT> children = new ArrayList<>();

        for (int childCount = 0; childCount < count; childCount++) {
            // clone and mutate child
            NeatGenomeT childGenome = Genome.cloneGenome(parent.item.genome(), true);
            NeatMutationFunctions.mutate(
                    random,
                    mutationSettings,
                    childGenome,
                    currentGeneration,
                    context.innovationMap);

            SpeciesMemberT child = createSpeciesMember(parent.item, childGenome);
            children.add(child);
        }

        return children;
    }

    @Override
    protected List<SpeciesMemberT> createOffspringSexual(EvolutionContext context,
                                                         FitnessItem<SpeciesMemberT> parentA,
                                                         FitnessItem<SpeciesMemberT> parentB,
                                                         long currentGeneration,
                                                         int count) {
        return createOffspringSexual(getNeatEvolutionContext(context), parentA, parentB, currentGeneration, count);
    }

    protected abstract NeatGenomeT createNewGenomeFromTemplate(NeatGenomeT template);

    protected abstract NeatGenomeT createRandomGenome(RandomGenerator random, int inputCount, int outputCount);

    private List<SpeciesMemberT> createOffspringSexual(NeatEvolutionContext context,
                                                       FitnessItem<SpeciesMemberT> parentA,
                                                       FitnessItem<SpeciesMemberT> parentB,
                                                       long currentGeneration,
                                                       int count) {

        int parentAInputCount = parentA.item.genome().neuralNet.neurons.count(NeuronType.Input);
        int parentAOutputCount = parentA.item.genome().neuralNet.neurons.count(NeuronType.Output);

        int parentBInputCount = parentB.item.genome().neuralNet.neurons.count(NeuronType.Input);
        int parentBOutputCount = parentB.item.genome().neuralNet.neurons.count(NeuronType.Output);

        if (parentAInputCount != parentBInputCount) {
            throw new IllegalArgumentException("Input count mismatch");
        }
        if (parentAOutputCount != parentBOutputCount) {
            throw new IllegalArgumentException("Output count mismatch");
        }

        FitnessItem<NeatGenomeT> parentAItem = createGenomeItem(parentA);
        FitnessItem<NeatGenomeT> parentBItem = createGenomeItem(parentB);

        GenomeParents<NeatGenomeT> parents = new GenomeParents<>(parentAItem, parentBItem);

        List<SpeciesMemberT> children = new ArrayList<>();

        for (int childCount = 0; childCount < count; childCount++) {

            // crossover
            NeatGenomeT childGenome = NeatCrossoverFunctions.crossover(
                    random,
                    crossoverSettings,
                    parents,
                    this::createNewGenomeFromTemplate);

            if (RandomFunctions.testProbability(random, mutationSettings.getSexualMutationProbability())) {
                // Mutate
                NeatMutationFunctions.mutate(
                        random,
                        mutationSettings,
                        childGenome,
                        currentGeneration,
                        context.innovationMap);
            }

            GenomeParents<SpeciesMemberT> memberParents = new GenomeParents<>(parentA, parentB);
            SortedGenomeParents<SpeciesMemberT> sortedMemberParents = memberParents.getSorted(random);

            SpeciesMemberT child = createSpeciesMember(sortedMemberParents.fittest.item, childGenome);
            children.add(child);
        }

        return children;
    }

    private FitnessItem<NeatGenomeT> createGenomeItem(FitnessItem<SpeciesMemberT> member) {
        return new FitnessItem<>(member.item.genome(), member.fitness);
    }

    // TODO: do this better
    private class NeatEvolutionContext extends EvolutionContext {

        public final Map<Long, Map<Long, Long>> innovationMap;

        public NeatEvolutionContext(EvolutionParameters evolutionParameters,
                                    SelectionStrategy selectionStrategy) {

            super(evolutionParameters, selectionStrategy);
            innovationMap = new HashMap<>();
        }
    }
}
