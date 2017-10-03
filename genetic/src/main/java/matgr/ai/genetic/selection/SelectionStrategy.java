package matgr.ai.genetic.selection;

import matgr.ai.genetic.FitnessItem;
import matgr.ai.genetic.PopulationFitnessSnapshot;
import matgr.ai.genetic.SpeciesMembersFitnessSnapshot;
import matgr.ai.math.DiscreteDistribution;
import matgr.ai.math.SortDirection;
import org.apache.commons.math3.random.RandomGenerator;

import java.util.ArrayList;
import java.util.List;
import java.util.UUID;

public abstract class SelectionStrategy {

    protected SelectionStrategy() {
    }

    public abstract boolean useGroupedSelectionSampling();

    @FunctionalInterface
    public interface MemberFilter {
        boolean test(UUID speciesId, FitnessItem<UUID> memberId);
    }

    public DiscreteDistribution<FitnessItem<UUID>> getSelectionDistribution(
            SpeciesMembersFitnessSnapshot fitnessSnapshot,
            MemberFilter memberFilter) {

        List<SpeciesMembersFitnessSnapshot> snapshots = new ArrayList<>();
        snapshots.add(fitnessSnapshot);

        return getSelectionDistribution(snapshots, memberFilter);
    }

    public DiscreteDistribution<FitnessItem<UUID>> getSelectionDistribution(
            PopulationFitnessSnapshot fitnessSnapshot,
            MemberFilter memberFilter) {

        return getSelectionDistribution(fitnessSnapshot.species, memberFilter);
    }

    public DiscreteDistribution<FitnessItem<UUID>> getSelectionDistribution(
            Iterable<SpeciesMembersFitnessSnapshot> fitnessSnapshots,
            MemberFilter memberFilter) {

        if (null == memberFilter) {
            memberFilter = (s, g) -> true;
        }

        return createSelectionDistribution(fitnessSnapshots, memberFilter);
    }

    public <T> List<T> select(RandomGenerator random, DiscreteDistribution<T> fitnessDistribution, int count) {

        List<T> results = trySelect(random, fitnessDistribution, count);

        if (results.size() != count) {
            throw new IllegalStateException(String.format("Failed to select %d genomes", count));
        }

        return results;
    }

    public <T> List<T> trySelect(RandomGenerator random, DiscreteDistribution<T> fitnessDistribution, int count) {

        List<T> results = new ArrayList<>();

        // TODO: do this better

        for (int i = 0; i < count; i++) {
            if ((fitnessDistribution.stats.count < 1) || (fitnessDistribution.groupedStats.count < 1)) {
                break;
            }

            T selected = fitnessDistribution.sample(random, useGroupedSelectionSampling());
            results.add(selected);

            if ((fitnessDistribution.stats.count < 2) || (fitnessDistribution.groupedStats.count < 2)) {
                break;
            }

            fitnessDistribution = fitnessDistribution.removeOutcome(selected);
        }

        return results;
    }

    protected abstract DiscreteDistribution<FitnessItem<UUID>> createSelectionDistribution(
            Iterable<SpeciesMembersFitnessSnapshot> fitnessSnapshots,
            MemberFilter memberFilter);

    protected DiscreteDistribution<FitnessItem<UUID>> createEvenProbabilityDistribution(
            Iterable<SpeciesMembersFitnessSnapshot> fitnessSnapshots,
            MemberFilter memberFilter) {

        List<FitnessItem<UUID>> items = new ArrayList<>();

        for (SpeciesMembersFitnessSnapshot fitnessSnapshot : fitnessSnapshots) {

            for (FitnessItem<UUID> member : fitnessSnapshot.genomesDescendingFitness) {

                if (memberFilter.test(fitnessSnapshot.speciesId(), member)) {
                    FitnessItem<UUID> genomeFitnessItem = new FitnessItem<>(member.item, 1.0);
                    items.add(genomeFitnessItem);
                }

            }
        }

        return DiscreteDistribution.create(items);
    }

    protected DiscreteDistribution<FitnessItem<UUID>> createBestOnlyProbabilityDistribution(
            Iterable<SpeciesMembersFitnessSnapshot> fitnessSnapshots,
            MemberFilter memberFilter) {

        List<FitnessItem<UUID>> items = new ArrayList<>();

        for (SpeciesMembersFitnessSnapshot fitnessSnapshot : fitnessSnapshots) {

            for (FitnessItem<UUID> member : fitnessSnapshot.genomesDescendingFitness) {

                if (memberFilter.test(fitnessSnapshot.speciesId(), member)) {

                    FitnessItem<UUID> genomeFitnessItem = new FitnessItem<>(member.item, 1.0);
                    items.add(genomeFitnessItem);
                    break;
                }
            }
        }

        return DiscreteDistribution.create(items);

    }

    protected List<FitnessItem<UUID>> filterAllMembers(Iterable<SpeciesMembersFitnessSnapshot> fitnessSnapshots,
                                                       MemberFilter memberFilter) {
        return filterAllMembers(fitnessSnapshots, memberFilter, null);
    }

    protected List<FitnessItem<UUID>> filterAllMembers(Iterable<SpeciesMembersFitnessSnapshot> fitnessSnapshots,
                                                       MemberFilter memberFilter,
                                                       SortDirection sortDirection) {

        List<FitnessItem<UUID>> filteredList = new ArrayList<>();

        for (SpeciesMembersFitnessSnapshot fitnessSnapshot : fitnessSnapshots) {

            List<FitnessItem<UUID>> members;

            switch (sortDirection) {
                case Ascending:
                    members = fitnessSnapshot.genomesAscendingFitness;
                    break;
                case Descending:
                    members = fitnessSnapshot.genomesDescendingFitness;
                    break;
                default:
                    members = fitnessSnapshot.genomes;
                    break;
            }

            for (FitnessItem<UUID> member : members) {

                if (memberFilter.test(fitnessSnapshot.speciesId(), member)) {
                    filteredList.add(member);
                }

            }
        }

        return filteredList;

    }
}
