package matgr.ai.neatsample.neat;

import matgr.ai.genetic.EvolutionContext;
import matgr.ai.math.clustering.Cluster;
import matgr.ai.neat.NeatGeneticAlgorithm;
import matgr.ai.neatsample.minesweepers.MineField;
import matgr.ai.neatsample.minesweepers.MineSweeperSettings;
import org.apache.commons.math3.random.RandomGenerator;

import java.util.List;

public class NeatMineSweeperGeneticAlgorithm extends NeatGeneticAlgorithm<
        NeatMineSweeperPopulation,
        NeatMineSweeperSpecies,
        NeatMineSweeper,
        NeatMineSweeperGenome> {

    public final MineSweeperSettings settings;

    public NeatMineSweeperGeneticAlgorithm(RandomGenerator random, MineSweeperSettings settings) {
        super(
                random,
                settings.getNeatCrossoverSettings(),
                settings.getNeatMutationSettings(),
                settings.getSpeciationStrategy());

        this.settings = settings;
    }

    @Override
    protected NeatMineSweeperPopulation createPopulation(EvolutionContext context,
                                                         List<NeatMineSweeperSpecies> species,
                                                         long generation) {
        return new NeatMineSweeperPopulation(species, generation);
    }

    @Override
    protected NeatMineSweeperSpecies createSpecies(Cluster<NeatMineSweeper> speciesMembers) {
        return new NeatMineSweeperSpecies(speciesMembers);
    }

    @Override
    protected NeatMineSweeper createSpeciesMember(NeatMineSweeper template, NeatMineSweeperGenome genome) {

        if (template == null) {
            return new NeatMineSweeper(random, genome, settings);
        }

        return new NeatMineSweeper(random, genome, settings);
    }

    @Override
    protected NeatMineSweeperGenome createNewGenome(int inputCount, int outputCount, double activationResponse) {
        MineField mineField = new MineField(random, settings);
        return new NeatMineSweeperGenome(mineField, inputCount, outputCount, activationResponse);
    }


}
