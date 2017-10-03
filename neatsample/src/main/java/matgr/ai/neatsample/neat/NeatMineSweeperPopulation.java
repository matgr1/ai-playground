package matgr.ai.neatsample.neat;

import matgr.ai.genetic.Population;

import java.util.List;

public class NeatMineSweeperPopulation extends Population<NeatMineSweeperSpecies> {

    public NeatMineSweeperPopulation(List<NeatMineSweeperSpecies> species, long generation) {
        super(species, generation);
    }
}
