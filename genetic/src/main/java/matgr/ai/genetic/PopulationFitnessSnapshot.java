package matgr.ai.genetic;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.UUID;

public class PopulationFitnessSnapshot extends GenomeListFitnessSnapshot {

    private final List<SpeciesMembersFitnessSnapshot> speciesMembersFitnessSnapshots;

    public final long generation;

    public final List<SpeciesMembersFitnessSnapshot> species;

    private PopulationFitnessSnapshot(
            long generation,
            Iterable<FitnessItem<UUID>> allMembers,
            List<SpeciesMembersFitnessSnapshot> speciesMembersFitnessSnapshots) {

        super(allMembers);

        this.generation = generation;

        this.speciesMembersFitnessSnapshots = speciesMembersFitnessSnapshots;
        this.species = Collections.unmodifiableList(this.speciesMembersFitnessSnapshots);
    }

    public static PopulationFitnessSnapshot create(Population<?> population) {

        List<FitnessItem<UUID>> allMembers = new ArrayList<>();
        List<SpeciesMembersFitnessSnapshot> speciesSnapshots = new ArrayList<>();

        for (Species<?> species : population.species()) {

            List<FitnessItem<UUID>> speciesMembers = new ArrayList<>();

            for (SpeciesMember member : species.members()) {

                double fitness = member.computeFitness();

                FitnessItem<UUID> fitnessItem = new FitnessItem<>(member.genome().genomeId(), fitness);

                speciesMembers.add(fitnessItem);
                allMembers.add(fitnessItem);
            }

            SpeciesMembersFitnessSnapshot speciesSnapshot = new SpeciesMembersFitnessSnapshot(
                    species.speciesId(),
                    speciesMembers);

            speciesSnapshots.add(speciesSnapshot);
        }

        return new PopulationFitnessSnapshot(population.generation(), allMembers, speciesSnapshots);

    }

    public int getSpeciesCount() {
        return speciesMembersFitnessSnapshots.size();
    }

    public SpeciesMembersFitnessSnapshot getSpeciesSnapshot(int index) {
        return species.get(index);
    }
}

