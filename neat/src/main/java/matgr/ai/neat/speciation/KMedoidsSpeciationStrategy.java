package matgr.ai.neat.speciation;

import matgr.ai.genetic.SpeciesMember;
import matgr.ai.neat.NeatGenome;
import matgr.ai.genetic.Population;
import matgr.ai.genetic.Species;
import matgr.ai.math.clustering.Cluster;
import matgr.ai.math.clustering.KMedoidClusteringAlgorithm;

import java.util.List;

public class KMedoidsSpeciationStrategy extends SpeciationStrategy {

    public final int initialClusterCount;

    public KMedoidsSpeciationStrategy(double excessFactor,
                                      double disjointFactor,
                                      double weightFactor,
                                      int initialClusterCount) {

        super(excessFactor, disjointFactor, weightFactor);
        this.initialClusterCount = initialClusterCount;
    }

    @Override
    public void reset() {
    }

    @Override
    protected <
            PopulationT extends Population<SpeciesT>,
            SpeciesT extends Species<SpeciesMemberT>,
            SpeciesMemberT extends SpeciesMember<? extends NeatGenome>>
    List<Cluster<SpeciesMemberT>> speciate(List<SpeciesMemberT> members, PopulationT previousPopulation) {

        NeatSpeciationClusteringAlgorithm<SpeciesMemberT> clusteringAlgorithm =
                new NeatSpeciationClusteringAlgorithm<>(this);

        if (null != previousPopulation) {
            List<List<SpeciesMemberT>> initialClusters = groupByPreviousRepresentatives(members, previousPopulation);
            return clusteringAlgorithm.refine(initialClusters);
        }

        return clusteringAlgorithm.compute(members, initialClusterCount);
    }

    private class NeatSpeciationClusteringAlgorithm<SpeciesMemberT extends SpeciesMember<? extends NeatGenome>>
            extends KMedoidClusteringAlgorithm<SpeciesMemberT> {

        public final KMedoidsSpeciationStrategy speciationStrategy;

        public NeatSpeciationClusteringAlgorithm(KMedoidsSpeciationStrategy speciationStrategy) {
            this.speciationStrategy = speciationStrategy;
        }

        @Override
        protected double computeDistance(SpeciesMemberT a, SpeciesMemberT b) {
            return NeatGenome.computeDistance(
                    a.genome(),
                    b.genome(),
                    speciationStrategy.excessFactor,
                    speciationStrategy.disjointFactor,
                    speciationStrategy.weightFactor);
        }
    }
}
