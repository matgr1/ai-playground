package matgr.ai.neatsample.neat;

import matgr.ai.neat.NeatGenome;
import matgr.ai.neatsample.minesweepers.MineField;
import matgr.ai.neatsample.minesweepers.MineSweeperGenome;
import matgr.ai.neuralnet.cyclic.OutputNeuronParameters;

import java.util.UUID;

public class NeatMineSweeperGenome extends NeatGenome implements MineSweeperGenome {

    private final MineField mineField;

    public NeatMineSweeperGenome(MineField mineField,
                                 int inputCount,
                                 int outputCount,
                                 double activationResponse) {
        super(inputCount, outputCount, activationResponse);
        this.mineField = mineField;
    }

    public NeatMineSweeperGenome(MineField mineField,
                                 int inputCount,
                                 Iterable<OutputNeuronParameters> outputNodesParameters,
                                 double activationResponse) {
        super(inputCount, outputNodesParameters, activationResponse);
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
