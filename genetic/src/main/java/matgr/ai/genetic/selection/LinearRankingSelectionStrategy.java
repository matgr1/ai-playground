package matgr.ai.genetic.selection;

import matgr.ai.genetic.FitnessItem;
import matgr.ai.genetic.SpeciesMembersFitnessSnapshot;
import matgr.ai.math.DiscreteDistribution;
import matgr.ai.math.SortDirection;

import java.util.ArrayList;
import java.util.List;
import java.util.UUID;

public class LinearRankingSelectionStrategy extends RankBasedSelectionStrategy {

    public LinearRankingSelectionStrategy(double selectivePressure) {
        super(selectivePressure);
    }

    @Override
    protected DiscreteDistribution<FitnessItem<UUID>> createSelectionDistribution(
            Iterable<SpeciesMembersFitnessSnapshot> fitnessSnapshots,
            MemberFilter memberFilter) {

        // NOTE: the SelectivePressure (SP) just controls the slope of the conversion: slope = SP/(1.0-SP)

        double selectivePressure = this.selectivePressure;

        if (selectivePressure == 0.0) {
            return createEvenProbabilityDistribution(fitnessSnapshots, memberFilter);
        }

        if (selectivePressure == 1.0) {
            return createBestOnlyProbabilityDistribution(fitnessSnapshots, memberFilter);
        }

        List<FitnessItem<UUID>> genomesAscendingFitness = filterAllMembers(
                fitnessSnapshots,
                memberFilter,
                SortDirection.Ascending);

        List<FitnessItem<UUID>> items = new ArrayList<>();

        double rankNormalizationFactor = 1.0 / (genomesAscendingFitness.size());
        double slope = selectivePressure / (1.0 - selectivePressure);

        double intercept = 1.0 - slope;

        for (int i = 0; i < genomesAscendingFitness.size(); i++) {
            FitnessItem<UUID> originalFitnessItem = genomesAscendingFitness.get(i);

            int rank = i + 1;
            double normalizedRank = rank * rankNormalizationFactor;

            double probability = Math.max(0.0, (slope * normalizedRank) + intercept);
            items.add(new FitnessItem<>(originalFitnessItem.item, probability));
        }

        return DiscreteDistribution.create(items);
    }

}
