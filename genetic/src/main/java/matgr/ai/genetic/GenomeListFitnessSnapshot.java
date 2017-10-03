package matgr.ai.genetic;

import matgr.ai.math.DiscreteDistributionItemSortComparator;
import matgr.ai.math.SortDirection;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;

public class GenomeListFitnessSnapshot {

    private final List<FitnessItem<UUID>> writableGenomes;
    private final Map<UUID, Double> genomeFitnessMap;

    public final List<FitnessItem<UUID>> genomes;
    public final List<FitnessItem<UUID>> genomesAscendingFitness;
    public final List<FitnessItem<UUID>> genomesDescendingFitness;

    protected GenomeListFitnessSnapshot(Iterable<FitnessItem<UUID>> genomes) {

        this.writableGenomes = new ArrayList<>();

        this.genomes = Collections.unmodifiableList(this.writableGenomes);

        SortedGenomesList sortedGenomesList = new SortedGenomesList();
        this.genomesAscendingFitness = sortedGenomesList.ascending;
        this.genomesDescendingFitness = sortedGenomesList.descending;

        this.genomeFitnessMap = new HashMap<>();

        for(FitnessItem<UUID> genome: genomes){
            writableGenomes.add(genome);
            genomeFitnessMap.put(genome.item, genome.fitness);
        }

        sortedGenomesList.initialize(genomes);
    }

    public int genomeCount() {
        return writableGenomes.size();
    }

    public double getFitness(Genome genome) {
        return genomeFitnessMap.get(genome.genomeId());
    }

    public boolean hasFitness(Genome genome) {
        return genomeFitnessMap.containsKey(genome.genomeId());
    }

    public double getFitness(UUID genomeId) {
        return genomeFitnessMap.get(genomeId);
    }

    public boolean hasFitness(UUID genomeId) {
        return genomeFitnessMap.containsKey(genomeId);
    }

    private class SortedGenomesList {

        private final List<FitnessItem<UUID>> writableAscending;
        private final List<FitnessItem<UUID>> writableDescending;

        public final List<FitnessItem<UUID>> ascending;
        public final List<FitnessItem<UUID>> descending;

        public SortedGenomesList() {
            writableAscending = new ArrayList<>();
            writableDescending = new ArrayList<>();

            ascending = Collections.unmodifiableList(writableAscending);
            descending = Collections.unmodifiableList(writableDescending);
        }

        void initialize(Iterable<FitnessItem<UUID>> genomes) {

            writableAscending.clear();
            writableDescending.clear();

            genomes.forEach(writableAscending::add);

            writableAscending.sort(new DiscreteDistributionItemSortComparator<>(SortDirection.Ascending));

            for (int i = writableAscending.size() - 1; i >= 0; i--) {
                writableDescending.add(writableAscending.get(i));
            }
        }
    }
}
