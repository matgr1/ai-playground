package matgr.ai.neat.speciation;

import matgr.ai.genetic.SpeciesMember;
import matgr.ai.neat.NeatGenome;
import matgr.ai.genetic.Population;
import matgr.ai.genetic.Species;
import matgr.ai.math.clustering.Cluster;

import java.util.ArrayList;
import java.util.List;

public abstract class SpeciationStrategy {

    @FunctionalInterface
    public interface CreateSpecies<
            SpeciesT extends Species<SpeciesMemberT>,
            SpeciesMemberT extends SpeciesMember<? extends NeatGenome>> {
        SpeciesT create(Cluster<SpeciesMemberT> species);
    }

    public double excessFactor;

    public double disjointFactor;

    public double weightFactor;

    protected SpeciationStrategy(double excessFactor, double disjointFactor, double weightFactor) {
        this.excessFactor = excessFactor;
        this.disjointFactor = disjointFactor;
        this.weightFactor = weightFactor;
    }

    public abstract void reset();

    public <
            SpeciesT extends Species<SpeciesMemberT>,
            SpeciesMemberT extends SpeciesMember<? extends NeatGenome>>
    List<SpeciesT> speciate(
            List<SpeciesMemberT> members,
            Population<SpeciesT> previousPopulation,
            CreateSpecies<SpeciesT, SpeciesMemberT> createSpecies) {

        List<Cluster<SpeciesMemberT>> speciatedMembers = speciate(members, previousPopulation);

        List<SpeciesT> species = new ArrayList<>();

        for (Cluster<SpeciesMemberT> speciesMembers : speciatedMembers) {
            SpeciesT curSpecies = createSpecies.create(speciesMembers);
            species.add(curSpecies);
        }

        return species;
    }

    protected abstract <
            PopulationT extends Population<SpeciesT>,
            SpeciesT extends Species<SpeciesMemberT>,
            SpeciesMemberT extends SpeciesMember<? extends NeatGenome>>
    List<Cluster<SpeciesMemberT>> speciate(
            List<SpeciesMemberT> members,
            PopulationT previousPopulation);

    protected <
            PopulationT extends Population<SpeciesT>,
            SpeciesT extends Species<SpeciesMemberT>,
            SpeciesMemberT extends SpeciesMember<? extends NeatGenome>>
    List<List<SpeciesMemberT>> groupByPreviousRepresentatives(
            List<SpeciesMemberT> members,
            PopulationT previousPopulation) {

        List<List<SpeciesMemberT>> initialClusters = new ArrayList<>();

        for (int i = 0; i < previousPopulation.speciesCount(); i++) {
            initialClusters.add(new ArrayList<>());
        }

        for (SpeciesMemberT member : members) {

            int minDistanceSpeciesIndex = -1;
            double minDistance = Double.NaN;

            for (int speciesIndex = 0; speciesIndex < previousPopulation.speciesCount(); speciesIndex++) {

                SpeciesT species = previousPopulation.getSpecies(speciesIndex);

                double distance = NeatGenome.computeDistance(
                        species.representative().genome(),
                        member.genome(),
                        excessFactor,
                        disjointFactor,
                        weightFactor);

                if (minDistanceSpeciesIndex < 0) {

                    minDistanceSpeciesIndex = speciesIndex;
                    minDistance = distance;

                } else {

                    if (distance < minDistance) {
                        minDistanceSpeciesIndex = speciesIndex;
                        minDistance = distance;
                    }

                }

            }

            List<SpeciesMemberT> minDistanceCluster = initialClusters.get(minDistanceSpeciesIndex);
            minDistanceCluster.add(member);

        }

        return initialClusters;
    }
}
