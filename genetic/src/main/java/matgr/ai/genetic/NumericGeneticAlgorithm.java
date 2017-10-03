package matgr.ai.genetic;

import matgr.ai.genetic.crossover.CrossoverFunctions;
import matgr.ai.genetic.mutation.MutationFunctions;
import matgr.ai.math.RandomFunctions;
import matgr.ai.math.clustering.Cluster;
import org.apache.commons.math3.random.RandomGenerator;

import java.util.ArrayList;
import java.util.List;
import java.util.UUID;

public abstract class NumericGeneticAlgorithm<
        PopulationT extends Population<SpeciesT>,
        SpeciesT extends Species<SpeciesMemberT>,
        SpeciesMemberT extends SpeciesMember<GenomeT>,
        GenomeT extends NumericGenome>
        extends GeneticAlgorithm<PopulationT, SpeciesT, SpeciesMemberT, GenomeT> {

    public final NumericCrossoverSettings crossoverSettings;

    public final NumericMutationSettings mutationSettings;

    protected NumericGeneticAlgorithm(RandomGenerator random,
                                      NumericCrossoverSettings crossoverSettings,
                                      NumericMutationSettings mutationSettings) {

        super(random);

        if (null == crossoverSettings) {
            throw new IllegalArgumentException("crossoverSettings not provided");
        }
        if (null == mutationSettings) {
            throw new IllegalArgumentException("mutationSettings not provided");
        }

        this.crossoverSettings = crossoverSettings;
        this.mutationSettings = mutationSettings;

    }

    @Override
    protected List<SpeciesT> speciate(EvolutionContext context,
                                      List<SpeciesMemberT> members,
                                      PopulationT previousPopulation) {

        List<SpeciesMemberT> newMembers = new ArrayList<>();

        newMembers.addAll(members);

        // TODO: pick a real representative? maybe the median? does it matter?
        SpeciesMemberT representative = RandomFunctions.selectItem(random, newMembers);

        List<SpeciesT> result = new ArrayList<>();
        result.add(createSpecies(new Cluster<>(newMembers, representative)));

        return result;

    }

    @Override
    protected List<SpeciesMemberT> createOffspringAsexual(EvolutionContext context,
                                                          FitnessItem<SpeciesMemberT> parent,
                                                          long currentGeneration,
                                                          int count) {
        List<SpeciesMemberT> children = new ArrayList<>();

        for (int childCount = 0; childCount < count; childCount++) {

            GenomeT childGenome = Genome.cloneGenome(parent.item.genome(), true);
            mutate(childGenome, currentGeneration);

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

        int genomeLength = parentA.item.genome().geneCount();

        if (genomeLength != parentB.item.genome().geneCount()) {
            throw new IllegalArgumentException("Genome length mismatch");
        }

        FitnessItem<SpeciesMemberT> parentAItem = new FitnessItem<>(parentA.item, parentA.fitness);
        FitnessItem<SpeciesMemberT> parentBItem = new FitnessItem<>(parentB.item, parentB.fitness);

        GenomeParents<SpeciesMemberT> parents = new GenomeParents<>(parentAItem, parentBItem);

        List<SpeciesMemberT> children = new ArrayList<>();

        for (int childCount = 0; childCount < count; childCount++) {

            SortedGenomeParents<SpeciesMemberT> sortedParents = parents.getSorted(random);

            // crossover
            List<Double> childGenes = new ArrayList<>();

            for (int i = 0; i < genomeLength; i++) {
                double fittestParentGene = sortedParents.fittest.item.genome().getGene(i);
                double otherParentGene = sortedParents.other.item.genome().getGene(i);

                if (RandomFunctions.testProbability(random, crossoverSettings.getProbability())) {

                    double childGene = CrossoverFunctions.crossover(
                            random,
                            crossoverSettings,
                            fittestParentGene,
                            otherParentGene);

                    childGenes.add(childGene);

                } else {

                    childGenes.add(fittestParentGene);

                }
            }

            GenomeT childGenome = createNewGenome(childGenes, UUID.randomUUID());

            // mutate
            if (RandomFunctions.testProbability(random, mutationSettings.getProbability())) {
                mutate(childGenome, currentGeneration);
            }

            SpeciesMemberT child = createSpeciesMember(sortedParents.fittest.item, childGenome);
            children.add(child);
        }

        return children;

    }

    protected abstract GenomeT createNewGenome(List<Double> genes, UUID genomeId);

    private void mutate(GenomeT genome, long currentGeneration) {

        int geneIndex = RandomFunctions.randomIndex(random, 0, genome.geneCount() - 1);

        double oldValue = genome.getGene(geneIndex);
        double newValue = MutationFunctions.mutate(random, mutationSettings, currentGeneration, oldValue);

        genome.setGene(geneIndex, newValue);

    }
}
