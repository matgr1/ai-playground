package matgr.ai.neatsample.neat;

import matgr.ai.neat.NeatGenome;
import matgr.ai.neatsample.minesweepers.MineField;
import matgr.ai.neatsample.minesweepers.MineSweeperGenome;
import matgr.ai.neuralnet.cyclic.NeuronParameters;

import java.util.UUID;

public class NeatMineSweeperGenome extends NeatGenome implements MineSweeperGenome {

    private final MineField mineField;

    public NeatMineSweeperGenome(MineField mineField,
                                 int inputCount,
                                 Iterable<NeuronParameters> outputNodesParameters) {
        super(inputCount, outputNodesParameters);
        this.mineField = mineField;
    }

    public NeatMineSweeperGenome(NeatMineSweeperGenome other) {
        super(other);
        this.mineField = other.mineField;
    }

    @Override
    public NeatMineSweeperGenome deepClone(UUID genomeId) {
        NeatMineSweeperGenome clone = deepClone();
        clone.genomeId = genomeId;
        return clone;
    }

    @Override
    public NeatMineSweeperGenome deepClone() {
        return new NeatMineSweeperGenome(this);
    }

    @Override
    public MineField getMineField() {
        return mineField;
    }
}
