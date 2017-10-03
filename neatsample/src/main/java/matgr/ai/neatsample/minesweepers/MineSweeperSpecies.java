package matgr.ai.neatsample.minesweepers;

import matgr.ai.genetic.Species;
import matgr.ai.math.clustering.Cluster;

public class MineSweeperSpecies<
        MineSweeperT extends MineSweeper<GenomeT>,
        GenomeT extends MineSweeperGenome>
        extends Species<MineSweeperT> {

    public MineSweeperSpecies(Cluster<MineSweeperT> species) {
        super(species);
    }
}

