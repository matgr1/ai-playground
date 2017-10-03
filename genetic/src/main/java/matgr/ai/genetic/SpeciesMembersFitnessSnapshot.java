package matgr.ai.genetic;

import java.util.UUID;

public class SpeciesMembersFitnessSnapshot extends GenomeListFitnessSnapshot {

    private final UUID speciesId;

    public SpeciesMembersFitnessSnapshot(UUID speciesId, Iterable<FitnessItem<UUID>> genomes) {
        super(genomes);
        this.speciesId = speciesId;
    }

    public UUID speciesId(){
        return speciesId;
    }

    public double getFitness(SpeciesMember member) {
        return getFitness(member.genome().genomeId());
    }

    public boolean hasFitness(SpeciesMember member) {
        return hasFitness(member.genome().genomeId());
    }
}
