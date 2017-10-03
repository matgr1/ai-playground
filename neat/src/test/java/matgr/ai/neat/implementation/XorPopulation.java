package matgr.ai.neat.implementation;

import matgr.ai.genetic.Population;

import java.util.List;

public class XorPopulation extends Population<XorSpecies> {

    protected XorPopulation(List<XorSpecies> species, long generation) throws IllegalArgumentException {
        super(species, generation);
    }
}
