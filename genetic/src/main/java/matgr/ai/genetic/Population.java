package matgr.ai.genetic;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;

public abstract class Population<SpeciesT extends Species<?>> {

    private final long generation;
    private final List<SpeciesT> species;

    private Map<UUID, SpeciesT> speciesMap;
    private Map<UUID, SpeciesT> speciesGenomeMap;

    protected Population(List<SpeciesT> species, long generation) throws IllegalArgumentException {

        this.generation = generation;

        this.species = new ArrayList<>();

        this.speciesMap = new HashMap<>();
        this.speciesGenomeMap = new HashMap<>();

        for (SpeciesT s : species) {

            this.speciesMap.put(s.speciesId(), s);
            this.species.add(s);

            for (SpeciesMember m : s.members()) {

                this.speciesGenomeMap.put(m.genome().genomeId(), s);
            }
        }
    }

    public long generation() {
        return generation;
    }

    public int speciesCount() {
        return species.size();
    }

    public SpeciesT getSpecies(int index) {
        return species.get(index);
    }

    public boolean hasSpecies(UUID speciesId) {
        return speciesMap.containsKey(speciesId);
    }

    public SpeciesT getSpecies(UUID speciesId) {
        return speciesMap.get(speciesId);
    }

    public boolean hasGenome(UUID genomeId) {
        return speciesGenomeMap.containsKey(genomeId);
    }

    public SpeciesT getGenomeSpecies(UUID genomeId) {
        return speciesGenomeMap.get(genomeId);
    }

    public Iterable<SpeciesT> species() {
        return species;
    }

    public int countGenomes() {

        int count = 0;

        for (Species species : species()) {
            count += species.memberCount();
        }

        return count;
    }
}
