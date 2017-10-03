package matgr.ai.neatsample.neat;

import matgr.ai.math.clustering.Cluster;
import matgr.ai.neatsample.minesweepers.MineSweeperSpecies;

public class NeatMineSweeperSpecies
        extends MineSweeperSpecies<NeatMineSweeper, NeatMineSweeperGenome> {

    public NeatMineSweeperSpecies(Cluster<NeatMineSweeper> species) {
        super(species);
    }
}
