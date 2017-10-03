package matgr.ai.genetic;

import matgr.ai.genetic.selection.SelectionStrategy;
import matgr.ai.math.DiscreteDistribution;
import matgr.ai.math.clustering.Cluster;
import org.apache.commons.math3.random.RandomGenerator;

import java.util.ArrayList;
import java.util.List;
import java.util.UUID;

public abstract class GeneticAlgorithm<
        PopulationT extends Population<SpeciesT>,
        SpeciesT extends Species<SpeciesMemberT>,
        SpeciesMemberT extends SpeciesMember<GenomeT>,
        GenomeT extends Genome> {

    public final RandomGenerator random;

    protected GeneticAlgorithm(RandomGenerator random) {
        this.random = random;
    }

    public GeneticAlgorithmResult<SpeciesMemberT> solve(
            List<SpeciesMemberT> initialPopulation,
            EvolutionParameters evolutionParameters,
            SelectionStrategy selectionStrategy,
            int maxIterations,
            double minFitness) {

        if (null == initialPopulation) {
            throw new IllegalArgumentException("initialPopulation not provided");
        }

        EvolutionContext context = createEvolutionContext(evolutionParameters, selectionStrategy);

        PopulationT population = createNewPopulation(context, initialPopulation);

        int iterations = 0;

        double bestMatchFitness = 0.0;
        UUID bestMatchGenomeId = null;
        boolean success = false;

        while ((maxIterations <= 0) || (iterations < maxIterations)) {
            if (Thread.currentThread().isInterrupted()) {
                break;
            }

            PopulationFitnessSnapshot fitnessSnapshot = PopulationFitnessSnapshot.create(population);

            iterations++;

            for (SpeciesMembersFitnessSnapshot species : fitnessSnapshot.species) {
                for (FitnessItem<UUID> member : species.genomes) {
                    if (Thread.currentThread().isInterrupted()) {
                        break;
                    }

                    if (member.fitness > bestMatchFitness) {
                        bestMatchFitness = member.fitness;
                        bestMatchGenomeId = member.item;
                    }
                }
            }

            if (bestMatchFitness >= minFitness) {
                success = true;
                break;
            }

            population = evolve(context, population, fitnessSnapshot);
        }

        SpeciesMemberT bestMatch = null;

        if (bestMatchGenomeId != null) {
            SpeciesT bestMatchSpecies = population.getGenomeSpecies(bestMatchGenomeId);
            bestMatch = bestMatchSpecies.getMember(bestMatchGenomeId);
        }

        return new GeneticAlgorithmResult<>(new FitnessItem<>(bestMatch, bestMatchFitness), iterations, success);
    }

    public PopulationT evolve(EvolutionContext context, PopulationT population) {
        PopulationFitnessSnapshot fitnessSnapshot = PopulationFitnessSnapshot.create(population);
        return evolve(context, population, fitnessSnapshot);
    }

    public PopulationT createNewPopulation(EvolutionContext context, List<SpeciesMemberT> members) {
        List<SpeciesT> species = speciate(context, members, null);
        return createPopulation(context, species, 0);
    }

    public EvolutionContext createEvolutionContext(EvolutionParameters evolutionParameters,
                                                   SelectionStrategy selectionStrategy) {

        return new EvolutionContext(evolutionParameters, selectionStrategy);

    }

    protected abstract List<SpeciesT> speciate(EvolutionContext context,
                                               List<SpeciesMemberT> members,
                                               PopulationT previousPopulation);

    protected abstract List<SpeciesMemberT> createOffspringAsexual(EvolutionContext context,
                                                                   FitnessItem<SpeciesMemberT> parent,
                                                                   long currentGeneration,
                                                                   int count);

    protected abstract List<SpeciesMemberT> createOffspringSexual(EvolutionContext context,
                                                                  FitnessItem<SpeciesMemberT> parentA,
                                                                  FitnessItem<SpeciesMemberT> parentB,
                                                                  long currentGeneration,
                                                                  int count);

    protected abstract PopulationT createPopulation(EvolutionContext context, List<SpeciesT> species, long generation);

    protected abstract SpeciesT createSpecies(Cluster<SpeciesMemberT> speciesMembers);

    protected abstract SpeciesMemberT createSpeciesMember(SpeciesMemberT template, GenomeT genome);

    private PopulationT evolve(EvolutionContext context,
                               PopulationT population,
                               PopulationFitnessSnapshot populationFitnessSnapshot) {

        int populationCount = population.countGenomes();

        if (populationCount < 2) {
            throw new IllegalArgumentException("Population must have at least 2 genomes");
        }

        int speciesCount = populationFitnessSnapshot.getSpeciesCount();

        List<SpeciesMemberT> nextPopulationSpecies = new ArrayList<>();

        double interspeciesProportion = context.evolutionParameters.getInterSpeciesSexualReproductionProportion();
        if (speciesCount <= 1) {
            interspeciesProportion = 0.0;
        }

        double proportionsTotal = context.evolutionParameters.getEliteProportion() +
                context.evolutionParameters.getAsexualReproductionProportion() +
                context.evolutionParameters.getSexualReproductionProportion() +
                interspeciesProportion;

        double eliteFraction = context.evolutionParameters.getEliteProportion() / proportionsTotal;
        double asexualFraction = context.evolutionParameters.getAsexualReproductionProportion() / proportionsTotal;
        double sexualFraction = context.evolutionParameters.getSexualReproductionProportion() / proportionsTotal;
        double interspeciesSexualFraction = interspeciesProportion / proportionsTotal;

        // TODO: Parallelize stuff (at least the species, there's no need to do them sequentially)

        double totalNormalizedSpeciesFitness = 0.0;
        double[] normalizedSpeciesTotalFitnesses = new double[speciesCount];

        for (int speciesIndex = 0; speciesIndex < speciesCount; speciesIndex++) {

            SpeciesMembersFitnessSnapshot speciesSnapshot = populationFitnessSnapshot.getSpeciesSnapshot(speciesIndex);

            double normalizedSpeciesFitness = getNormalizedSpeciesTotalFitness(speciesSnapshot);
            normalizedSpeciesTotalFitnesses[speciesIndex] = normalizedSpeciesFitness;

            totalNormalizedSpeciesFitness += normalizedSpeciesFitness;
        }

        int[] targetSpeciesSizes = new int[speciesCount];

        int totalOutputSize = 0;
        int bestSpeciesIndex = -1;
        double bestSpeciesFitness = Double.NEGATIVE_INFINITY;

        for (int speciesIndex = 0; speciesIndex < speciesCount; speciesIndex++) {
            double normalizedSpeciesFitness = normalizedSpeciesTotalFitnesses[speciesIndex];
            double speciesSizeProportion = normalizedSpeciesFitness / totalNormalizedSpeciesFitness;

            int curTargetSize = (int) Math.floor(speciesSizeProportion * populationCount);

            targetSpeciesSizes[speciesIndex] = curTargetSize;
            totalOutputSize += curTargetSize;

            // TODO: should some of this be by average instead of total?
            if (normalizedSpeciesFitness > bestSpeciesFitness) {
                bestSpeciesIndex = speciesIndex;
                bestSpeciesFitness = normalizedSpeciesFitness;
            }
        }

        if (totalOutputSize < populationCount) {
            // put the extra in the best species...
            targetSpeciesSizes[bestSpeciesIndex] += populationCount - totalOutputSize;
        }

        DiscreteDistribution<FitnessItem<UUID>>[] selectionDistributions =
                DiscreteDistribution.createArray(speciesCount);

        DiscreteDistribution<FitnessItem<UUID>>[] otherSelectionDistributions =
                DiscreteDistribution.createArray(speciesCount);

        int[] interspeciesSexualCounts = new int[speciesCount];

        for (int speciesIndex = 0; speciesIndex < speciesCount; speciesIndex++) {

            SpeciesMembersFitnessSnapshot speciesSnapshot =
                    populationFitnessSnapshot.getSpeciesSnapshot(speciesIndex);

            int targetSpeciesSize = targetSpeciesSizes[speciesIndex];

            int eliteCount = (int) Math.floor(targetSpeciesSize * eliteFraction);
            int asexualCount = (int) Math.floor(targetSpeciesSize * asexualFraction);
            int sexualCount = (int) Math.floor(targetSpeciesSize * sexualFraction);
            int interspeciesSexualCount = (int) Math.floor(targetSpeciesSize * interspeciesSexualFraction);

            DiscreteDistribution<FitnessItem<UUID>> selectionDistribution =
                    context.selectionStrategy.getSelectionDistribution(speciesSnapshot, null);

            selectionDistributions[speciesIndex] = selectionDistribution;

            if (interspeciesSexualCount > 0) {
                interspeciesSexualCounts[speciesIndex] = interspeciesSexualCount;

                DiscreteDistribution<FitnessItem<UUID>> otherSelectionDistribution =
                        context.selectionStrategy.getSelectionDistribution(
                                populationFitnessSnapshot,
                                (speciesId, genome) -> speciesId != speciesSnapshot.speciesId());

                otherSelectionDistributions[speciesIndex] = otherSelectionDistribution;
            }

            int remaining = targetSpeciesSize - (eliteCount + asexualCount + sexualCount + interspeciesSexualCount);

            // add any extras to sexual reproduction
            sexualCount += remaining;

            // TODO: if targetSpeciesSize is too large for the current species size, then there's not a whole lot of
            //       diversity for offspring... this may be OK as long as there's still mutation?
            //		 maybe just increase the interspecies mating for these? (interspecies should be picking a
            //       different item from the entire pool not consisting of the current species (using the selection
            //       strategy passed in)...
            // TODO: don't let species mate if they haven't improved in a long time (if none have improved noticeably,
            //       just do a big mutation?)
            // TODO: need to control species sizes? maybe group them differently? (some kind of dictionary of
            //       innovationIDs-->connectionWeights? and a measure of the euclidean distance (or something) between
            //       these?)

            for (int i = 0; i < eliteCount; i++) {

                FitnessItem<UUID> eliteItem = populationFitnessSnapshot.genomesDescendingFitness.get(i);

                SpeciesT eliteMemberSpecies = population.getGenomeSpecies(eliteItem.item);
                SpeciesMemberT eliteMember = eliteMemberSpecies.getMember(eliteItem.item);

                for (int j = 0; j < context.evolutionParameters.getEliteCopies(); j++) {

                    GenomeT eliteGenome = Genome.cloneGenome(eliteMember.genome(), true);
                    SpeciesMemberT eliteClone = createSpeciesMember(eliteMember, eliteGenome);

                    nextPopulationSpecies.add(eliteClone);
                }

            }

            for (int i = 0; i < asexualCount; i++) {

                List<FitnessItem<UUID>> parents = context.selectionStrategy.select(
                        random,
                        selectionDistribution,
                        1);

                List<SpeciesMemberT> children = createOffspringAsexual(
                        context,
                        createMemberFitnessItem(population, parents.get(0)),
                        populationFitnessSnapshot.generation,
                        1);

                nextPopulationSpecies.add(children.get(0));

            }

            for (int i = 0; i < sexualCount; i++) {

                List<FitnessItem<UUID>> parents = context.selectionStrategy.trySelect(
                        random,
                        selectionDistribution,
                        2);

                List<SpeciesMemberT> children;

                if (parents.size() < 2) {

                    children = createOffspringAsexual(
                            context,
                            createMemberFitnessItem(population, parents.get(0)),
                            populationFitnessSnapshot.generation,
                            1);

                } else {

                    children = createOffspringSexual(
                            context,
                            createMemberFitnessItem(population, parents.get(0)),
                            createMemberFitnessItem(population, parents.get(1)),
                            populationFitnessSnapshot.generation,
                            1);

                }

                nextPopulationSpecies.add(children.get(0));

            }
        }

        // NOTE: running this loop separately since the interspecies stuff needs ALL selection distributions (and don't
        //       want to have to calculate them multiple times)
        for (int speciesIndex = 0; speciesIndex < speciesCount; speciesIndex++) {

            int interspeciesSexualCount = interspeciesSexualCounts[speciesIndex];

            DiscreteDistribution<FitnessItem<UUID>> selectionDistribution =
                    selectionDistributions[speciesIndex];

            for (int i = 0; i < interspeciesSexualCount; i++) {

                List<FitnessItem<UUID>> parentsA = context.selectionStrategy.select(
                        random,
                        selectionDistribution,
                        1);

                DiscreteDistribution<FitnessItem<UUID>> otherSelectionDistribution =
                        otherSelectionDistributions[speciesIndex];

                List<FitnessItem<UUID>> parentsB = context.selectionStrategy.select(
                        random,
                        otherSelectionDistribution,
                        1);

                List<SpeciesMemberT> children = createOffspringSexual(
                        context,
                        createMemberFitnessItem(population, parentsA.get(0)),
                        createMemberFitnessItem(population, parentsB.get(0)),
                        populationFitnessSnapshot.generation,
                        1);

                nextPopulationSpecies.add(children.get(0));

            }
        }

        // NOTE: sanity check
        if (nextPopulationSpecies.size() != populationCount) {
            throw new IllegalStateException("Failed to create correctly sized population");
        }

        List<SpeciesT> species = speciate(context, nextPopulationSpecies, population);

        return createPopulation(context, species, population.generation() + 1);
    }

    private FitnessItem<SpeciesMemberT> createMemberFitnessItem(PopulationT population,
                                                                FitnessItem<UUID> memberFitnessItem) {

        SpeciesT memberSpecies = population.getGenomeSpecies(memberFitnessItem.item);
        SpeciesMemberT member = memberSpecies.getMember(memberFitnessItem.item);

        return new FitnessItem<>(member, memberFitnessItem.fitness);
    }

    private double getNormalizedSpeciesTotalFitness(
            SpeciesMembersFitnessSnapshot fitnessSnapshot) {

        if (fitnessSnapshot.genomes.size() <= 0) {
            throw new IllegalArgumentException("Snapshot has no genomes");
        }

        double minFitness = Double.POSITIVE_INFINITY;
        for (FitnessItem<UUID> member : fitnessSnapshot.genomes) {
            minFitness = Math.min(minFitness, member.fitness);
        }

        double normalizationAmount = 0.0;
        if (minFitness < 0) {
            normalizationAmount = -minFitness;
        }

        double normalizedFitness = 0.0;

        for (FitnessItem<UUID> member : fitnessSnapshot.genomes) {
            double fitness = member.fitness + normalizationAmount;
            normalizedFitness += fitness;
        }

        normalizedFitness = normalizedFitness / (double) fitnessSnapshot.genomeCount();

        return normalizedFitness;

    }
}
