package matgr.ai.neatsample.neat;

import matgr.ai.neatsample.minesweepers.MineSweeper;
import matgr.ai.neatsample.minesweepers.MineSweeperSettings;
import org.apache.commons.math3.random.RandomGenerator;

import java.util.List;

public class NeatMineSweeper extends MineSweeper<NeatMineSweeperGenome> {

    public NeatMineSweeper(RandomGenerator random, NeatMineSweeperGenome genome, MineSweeperSettings settings) {
        super(random, genome, settings);
    }

    @Override
    protected List<Double> activateNeuralNet(List<Double> inputs, double bias) {

        // TODO: pass these in?
        final int maxStepsPerActivation = 10;
        final boolean resetStateBeforeActivation = false;

        return genome.neuralNet.activateSingle(inputs, bias, maxStepsPerActivation, resetStateBeforeActivation);
    }

    @Override
    public NeatMineSweeperGenome genome() {
        return genome;
    }

    @Override
    public double computeFitness() {
        return getFitness();
    }
}