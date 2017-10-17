package matgr.ai.neatsample.neat;

import matgr.ai.genetic.EvolutionContext;
import matgr.ai.math.clustering.Cluster;
import matgr.ai.neat.NeatGeneticAlgorithm;
import matgr.ai.neat.mutation.NeatMutationFunctions;
import matgr.ai.neatsample.minesweepers.MineField;
import matgr.ai.neatsample.minesweepers.MineSweeperSettings;
import matgr.ai.neuralnet.activation.ActivationFunction;
import matgr.ai.neuralnet.Neuron;
import matgr.ai.neuralnet.NeuronType;
import matgr.ai.neuralnet.cyclic.NeuronParameters;
import matgr.ai.neuralnet.cyclic.CyclicNeuron;
import org.apache.commons.math3.random.RandomGenerator;

import java.util.ArrayList;
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
    protected NeatMineSweeperGenome createNewGenomeFromTemplate(NeatMineSweeperGenome template) {

        int inputCount = template.neuralNet.neurons.count(NeuronType.Input);
        Iterable<CyclicNeuron> outputNeurons = template.neuralNet.neurons.values(NeuronType.Output);

        List<NeuronParameters> outputParameters = new ArrayList<>();

        for (CyclicNeuron outputNeuron : outputNeurons) {

            ActivationFunction activationFunction = outputNeuron.getActivationFunction();
            double[] activationFunctionParameters = outputNeuron.getActivationFunctionParameters();

            NeuronParameters parameters = new NeuronParameters(
                    activationFunction,
                    activationFunctionParameters);

            outputParameters.add(parameters);
        }

        // TODO: clone the minefield from the template?
        MineField mineField = new MineField(random, settings);
        return new NeatMineSweeperGenome(mineField, inputCount, outputParameters);
    }

    @Override
    protected NeatMineSweeperGenome createRandomGenome(RandomGenerator random, int inputCount, int outputCount) {

        List<NeuronParameters> outputParameters = new ArrayList<>();

        for (int i = 0; i < outputCount; i++) {

            ActivationFunction activationFunction = NeatMutationFunctions.getRandomActivationFunction(random);
            NeuronParameters parameters = new NeuronParameters(
                    activationFunction,
                    activationFunction.defaultParameters());

            outputParameters.add(parameters);
        }

        MineField mineField = new MineField(random, settings);
        return new NeatMineSweeperGenome(mineField, inputCount, outputParameters);
    }

}
