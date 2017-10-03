package matgr.ai.genetic.selection;

import matgr.ai.genetic.FitnessItem;
import matgr.ai.genetic.SpeciesMembersFitnessSnapshot;
import matgr.ai.math.DiscreteDistribution;

import java.util.ArrayList;
import java.util.List;
import java.util.UUID;

public class RouletteWheelSelectionStrategy extends SelectionStrategy {

    private final boolean useGroupedSelectionSampling;

    public RouletteWheelSelectionStrategy(boolean groupedSelectionSampling) {
        this.useGroupedSelectionSampling = groupedSelectionSampling;
    }

    @Override
    public boolean useGroupedSelectionSampling() {
        return useGroupedSelectionSampling;
    }

    @Override
    protected DiscreteDistribution<FitnessItem<UUID>> createSelectionDistribution(
            Iterable<SpeciesMembersFitnessSnapshot> fitnessSnapshots,
            MemberFilter memberFilter) {

        double minFitness = Double.POSITIVE_INFINITY;

        for (SpeciesMembersFitnessSnapshot fitnessSnapshot : fitnessSnapshots) {

            for (FitnessItem<UUID> genome : fitnessSnapshot.genomes) {

                if (memberFilter.test(fitnessSnapshot.speciesId(), genome)) {
                    minFitness = Math.min(minFitness, genome.fitness);
                }

            }
        }

        List<FitnessItem<UUID>> items = new ArrayList<>();

        for (SpeciesMembersFitnessSnapshot fitnessSnapshot : fitnessSnapshots) {

            for (FitnessItem<UUID> genome : fitnessSnapshot.genomes) {

                if (memberFilter.test(fitnessSnapshot.speciesId(), genome)) {

                    double fitness = genome.fitness - minFitness;

                    items.add(new FitnessItem<>(genome.item, fitness));
                }
            }
        }

        if (items.size() <= 0) {
            throw new IllegalArgumentException("Snapshot has no genomes");
        }

        return DiscreteDistribution.create(items);

    }
}
