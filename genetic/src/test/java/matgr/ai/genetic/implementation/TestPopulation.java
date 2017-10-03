package matgr.ai.genetic.implementation;

import matgr.ai.genetic.Population;

import java.util.List;

public class TestPopulation extends Population<TestSpecies> {

    protected TestPopulation(List<TestSpecies> testSpecies, long generation)  {
        super(testSpecies, generation);
    }
}
