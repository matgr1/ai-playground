package matgr.ai.genetic.selection;

import matgr.ai.genetic.FitnessItem;
import matgr.ai.genetic.SpeciesMembersFitnessSnapshot;
import matgr.ai.math.DiscreteDistribution;
import matgr.ai.math.SortDirection;

import java.util.ArrayList;
import java.util.List;
import java.util.UUID;

public class ExponentialRankingSelectionStrategy extends RankBasedSelectionStrategy {

    public ExponentialRankingSelectionStrategy(double selectivePressure) {
        super(selectivePressure);
    }

    // TODO: not sure this works how it should? or maybe it's just that the control of the SelectivePressure parameter
    //       is difficult this way... (when everything's normalized)
    //		 check here: http://www.geatbx.com/docu/algindex-02.html
    //		 and here: http://www.pohlheim.com/Papers/mpga_gal95/gal2_3.html#Non-linear Ranking
    //		 and here: https://www.researchgate.net/publication/224645723_On_Nonlinear_Fitness_Functions_for_Ranking-Based_Selection
    //		 and here: http://www.tik.ee.ethz.ch/file/6c0e384dceb283cd4301339a895b72b8/TIK-Report11.pdf
    @Override
    protected DiscreteDistribution<FitnessItem<UUID>> createSelectionDistribution(
            Iterable<SpeciesMembersFitnessSnapshot> fitnessSnapshots,
            MemberFilter memberFilter) {

        // NOTE: this is from chapter 6 here (with some modifications):
        //       http://www4.hcmut.edu.vn/~hthoang/dktm/Selection.pdf
        //		 - rank is normalized to the range (0.0,1.0]
        //		 - "c" is reversed so that higher values mean higher "exponentiality"
        //		 - there is no normalization of the probabilities, since this is not required for DiscreteDistribution

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

        for (int i = 0; i < genomesAscendingFitness.size(); i++) {

            FitnessItem<UUID> originalFitnessItem = genomesAscendingFitness.get(i);

            int rank = i + 1;
            double normalizedRank = rank * rankNormalizationFactor;

            double probability = Math.pow((1.0 - selectivePressure), 1.0 - normalizedRank);

            items.add(new FitnessItem<>(originalFitnessItem.item, probability));
        }

        return DiscreteDistribution.create(items);
    }
}
