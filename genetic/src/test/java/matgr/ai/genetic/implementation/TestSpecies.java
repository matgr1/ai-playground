package matgr.ai.genetic.implementation;

import matgr.ai.genetic.Species;
import matgr.ai.math.clustering.Cluster;

public class TestSpecies extends Species<TestSpeciesMember> {
    protected TestSpecies(Cluster<TestSpeciesMember> species) {
        super(species);
    }
}
