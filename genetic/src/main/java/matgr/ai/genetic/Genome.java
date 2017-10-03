package matgr.ai.genetic;

import java.util.UUID;

public interface Genome {

    UUID genomeId();

    Genome deepClone(UUID genomeId);

    static <GenomeT extends Genome> GenomeT cloneGenome(GenomeT genome, boolean newId) {

        @SuppressWarnings("unchecked")
        GenomeT clone = (GenomeT) genome.deepClone(newId ? UUID.randomUUID() : genome.genomeId());

        if (!newId) {
            if (clone.genomeId() != genome.genomeId()) {
                throw new IllegalStateException("Invalid clone - Genome ID should not be altered by cloning");
            }
        }

        if (clone.getClass() != genome.getClass()) {
            throw new IllegalArgumentException("Invalid clone - deepClone not overridden correctly in derived class");
        }

        return clone;
    }

}
