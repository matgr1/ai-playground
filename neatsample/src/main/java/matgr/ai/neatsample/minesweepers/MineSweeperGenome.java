package matgr.ai.neatsample.minesweepers;

import matgr.ai.genetic.Genome;

public interface MineSweeperGenome extends Genome {
    MineField getMineField();
}
