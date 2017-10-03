package matgr.ai.neat.implementation;

import matgr.ai.genetic.Species;
import matgr.ai.math.clustering.Cluster;

public class XorSpecies extends Species<XorSpeciesMember> {

    protected XorSpecies(Cluster<XorSpeciesMember> species) {
        super(species);
    }

}
