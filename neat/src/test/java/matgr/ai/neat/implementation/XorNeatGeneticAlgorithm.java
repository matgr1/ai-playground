package matgr.ai.neat.implementation;

import matgr.ai.genetic.EvolutionContext;
import matgr.ai.math.clustering.Cluster;
import matgr.ai.neat.NeatGeneticAlgorithm;
import matgr.ai.neat.NeatGenome;
import matgr.ai.neat.crossover.NeatCrossoverSettings;
import matgr.ai.neat.mutation.NeatMutationFunctions;
import matgr.ai.neat.mutation.NeatMutationSettings;
import matgr.ai.neat.speciation.SpeciationStrategy;
import matgr.ai.neuralnet.activation.ActivationFunction;
import matgr.ai.neuralnet.cyclic.Neuron;
import matgr.ai.neuralnet.cyclic.NeuronType;
import matgr.ai.neuralnet.cyclic.OutputNeuronParameters;
import org.apache.commons.math3.random.RandomGenerator;

import java.util.ArrayList;
import java.util.List;

public class XorNeatGeneticAlgorithm
        extends NeatGeneticAlgorithm<XorPopulation, XorSpecies, XorSpeciesMember, NeatGenome> {

    public XorNeatGeneticAlgorithm(RandomGenerator random,
                                   NeatCrossoverSettings crossoverSettings,
                                   NeatMutationSettings mutationSettings,
                                   SpeciationStrategy speciationStrategy) {
        super(random, crossoverSettings, mutationSettings, speciationStrategy);
    }

    @Override
    protected NeatGenome createNewGenomeFromTemplate(NeatGenome template) {

        int inputCount = template.neuralNet.neurons.count(NeuronType.Input);
        Iterable<Neuron> outputNeurons = template.neuralNet.neurons.values(NeuronType.Output);

        List<OutputNeuronParameters> outputParameters = new ArrayList<>();

        for (Neuron outputNeuron : outputNeurons) {

            ActivationFunction activationFunction = outputNeuron.getActivationFunction();
            double[] activationFunctionParameters = outputNeuron.getActivationFunctionParameters();

            OutputNeuronParameters parameters = new OutputNeuronParameters(
                    activationFunction,
                    activationFunctionParameters);

            outputParameters.add(parameters);
        }

        return new NeatGenome(inputCount, outputParameters);
    }

    @Override
    protected NeatGenome createRandomGenome(RandomGenerator random, int inputCount, int outputCount) {

        List<OutputNeuronParameters> outputParameters = new ArrayList<>();

        for (int i = 0; i < outputCount; i++) {

            ActivationFunction activationFunction = NeatMutationFunctions.getRandomActivationFunction(random);
            OutputNeuronParameters parameters = new OutputNeuronParameters(
                    activationFunction,
                    activationFunction.defaultParameters());

            outputParameters.add(parameters);
        }

        return new NeatGenome(inputCount, outputParameters);
    }

    @Override
    protected XorPopulation createPopulation(EvolutionContext context, List<XorSpecies> species, long generation) {
        return new XorPopulation(species, generation);
    }

    @Override
    protected XorSpecies createSpecies(Cluster<XorSpeciesMember> speciesMembers) {
        return new XorSpecies(speciesMembers);
    }

    @Override
    protected XorSpeciesMember createSpeciesMember(XorSpeciesMember template, NeatGenome genome) {
        return new XorSpeciesMember(genome);
    }
}
