package matgr.ai.neat.speciation;

import matgr.ai.genetic.SpeciesMember;
import matgr.ai.neat.NeatGenome;
import matgr.ai.genetic.Population;
import matgr.ai.genetic.Species;
import matgr.ai.math.clustering.Cluster;

import java.util.ArrayList;
import java.util.List;

public class DefaultSpeciationStrategy extends SpeciationStrategy {

    public int baseGenomeSize;

    public int minGenomeNormalizationSize;

    public double distanceThreshold;

    public boolean useDynamicDistanceThreshold;

    public Long dynamicDistanceThresholdUpdateInterval;

    public int dynamicDistanceThresholdMinSpeciesCount;

    public int dynamicDistanceThresholdMaxSpeciesCount;

    public double dynamicDistanceThresholdMin;

    public double dynamicDistanceThresholdModifier;

    public double currentDistanceThreshold;

    public DefaultSpeciationStrategy(
            double excessFactor,
            double disjointFactor,
            double weightFactor,
            double distanceThreshold,
            int baseGenomeSize,
            int minGenomeNormalizationSize,
            boolean useDynamicDistanceThreshold,
            long dynamicDistanceThresholdUpdateInterval,
            int dynamicDistanceThresholdMinSpeciesCount,
            int dynamicDistanceThresholdMaxSpeciesCount,
            double dynamicDistanceThresholdModifier,
            double dynamicDistanceThresholdMin) {

        super(excessFactor, disjointFactor, weightFactor);
        this.distanceThreshold = distanceThreshold;
        this.baseGenomeSize = baseGenomeSize;
        this.minGenomeNormalizationSize = minGenomeNormalizationSize;

        this.useDynamicDistanceThreshold = useDynamicDistanceThreshold;
        this.dynamicDistanceThresholdUpdateInterval = dynamicDistanceThresholdUpdateInterval;
        this.dynamicDistanceThresholdMinSpeciesCount = dynamicDistanceThresholdMinSpeciesCount;
        this.dynamicDistanceThresholdMaxSpeciesCount = dynamicDistanceThresholdMaxSpeciesCount;
        this.dynamicDistanceThresholdModifier = dynamicDistanceThresholdModifier;
        this.dynamicDistanceThresholdMin = dynamicDistanceThresholdMin;

        resetCurrentDistanceThreshold();
    }

    @Override
    public void reset() {
        resetCurrentDistanceThreshold();
    }

    @Override
    protected <
            PopulationT extends Population<SpeciesT>,
            SpeciesT extends Species<SpeciesMemberT>,
            SpeciesMemberT extends SpeciesMember<? extends NeatGenome>>
    List<Cluster<SpeciesMemberT>> speciate(
            List<SpeciesMemberT> members,
            PopulationT previousPopulation) {

        List<Cluster<SpeciesMemberT>> speciatedMembers = speciateOnce(members, previousPopulation);

        if (useDynamicDistanceThreshold) {
            long speciationCount = 0;
            if (null != previousPopulation) {
                speciationCount = previousPopulation.generation();
            }

            if (0 == (speciationCount % dynamicDistanceThresholdUpdateInterval)) {
                double modificationAmount = 0.0;

                if (speciatedMembers.size() > dynamicDistanceThresholdMaxSpeciesCount) {
                    modificationAmount = dynamicDistanceThresholdModifier;
                }
                if (speciatedMembers.size() < dynamicDistanceThresholdMinSpeciesCount) {
                    modificationAmount = -dynamicDistanceThresholdModifier;
                }

                if (0.0 != modificationAmount) {
                    double newValue = currentDistanceThreshold + modificationAmount;
                    currentDistanceThreshold = Math.max(newValue, dynamicDistanceThresholdMin);

                    // try once more with the new threshold
                    speciatedMembers = speciateOnce(members, previousPopulation);
                }
            }
        }

        return speciatedMembers;
    }

    private void resetCurrentDistanceThreshold() {
        currentDistanceThreshold = distanceThreshold;
    }

    private <
            PopulationT extends Population<SpeciesT>,
            SpeciesT extends Species<SpeciesMemberT>,
            SpeciesMemberT extends SpeciesMember<? extends NeatGenome>>
    List<Cluster<SpeciesMemberT>> speciateOnce(
            List<SpeciesMemberT> members,
            PopulationT previousPopulation) {

        // NOTE: this is the default algorithm from here: http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf
        //       (Evolving Neural Networks through Augmenting Topologies)

        List<List<SpeciesMemberT>> speciatedMembers = new ArrayList<>();
        List<SpeciesMemberT> representativeMembers = new ArrayList<>();

        if (null != previousPopulation) {

            speciatedMembers = groupByPreviousRepresentatives(members, previousPopulation);

            for (List<SpeciesMemberT> speciesMembers : speciatedMembers) {

                SpeciesMemberT representative = null;
                NeatGenome representativeGenome = null;

                if (speciesMembers.size() > 0) {

                    for (SpeciesMemberT member : speciesMembers) {

                        if (isBetterRepresentativeThanCurrent(representativeGenome, member.genome())) {
                            representative = member;
                            representativeGenome = member.genome();
                        }
                    }
                }

                representativeMembers.add(representative);
            }

        } else {

            for (SpeciesMemberT member : members) {

                boolean foundSpecies = false;

                for (int speciesIndex = 0; speciesIndex < representativeMembers.size(); speciesIndex++) {

                    SpeciesMemberT representative = representativeMembers.get(speciesIndex);

                    double distance = NeatGenome.computeDistance(
                            member.genome(),
                            representative.genome(),
                            excessFactor,
                            disjointFactor,
                            weightFactor,
                            baseGenomeSize,
                            minGenomeNormalizationSize);

                    if (distance < currentDistanceThreshold) {

                        speciatedMembers.get(speciesIndex).add(member);

                        if (isBetterRepresentativeThanCurrent(representative.genome(), member.genome())) {
                            representativeMembers.set(speciesIndex, member);
                        }

                        foundSpecies = true;
                        break;
                    }
                }

                if (!foundSpecies) {

                    List<SpeciesMemberT> newList = new ArrayList<>();
                    newList.add(member);
                    speciatedMembers.add(newList);

                    representativeMembers.add(member);
                }
            }
        }

        List<Cluster<SpeciesMemberT>> results = new ArrayList<>();

        for (int i = 0; i < speciatedMembers.size(); i++) {

            List<SpeciesMemberT> clusterMembers = speciatedMembers.get(i);
            SpeciesMemberT clusterRepresentative = representativeMembers.get(i);

            Cluster<SpeciesMemberT> cluster = new Cluster<>(clusterMembers, clusterRepresentative);
            results.add(cluster);
        }

        return results;
    }

    private boolean isBetterRepresentativeThanCurrent(NeatGenome currentRepresentative, NeatGenome genome) {

        if (null == genome) {
            throw new IllegalArgumentException("genome not provided");
        }

        if (null == currentRepresentative) {
            return true;
        }

        // TODO: try other things here... maybe smallest? or use some other measure of distance?
        if (genome.neuralNet.connections.count() > currentRepresentative.neuralNet.connections.count()) {
            return true;
        }

        return false;
    }
}
