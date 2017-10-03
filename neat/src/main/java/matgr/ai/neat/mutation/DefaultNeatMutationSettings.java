package matgr.ai.neat.mutation;

import matgr.ai.genetic.mutation.MutationSettings;

import java.util.Map;

public class DefaultNeatMutationSettings implements NeatMutationSettings {

    public double sexualMutationProbability;

    public MutationSettings connectionWeightsMutationSettings;

    public Map<NeatStructuralMutationType, Double> mutationProbabilityProportions;

    public DefaultNeatMutationSettings(double sexualMutationProbability,
                                       MutationSettings connectionWeightsMutationSettings,
                                       Map<NeatStructuralMutationType, Double> mutationProbabilityProportions) {
        this.sexualMutationProbability = sexualMutationProbability;
        this.connectionWeightsMutationSettings = connectionWeightsMutationSettings;
        this.mutationProbabilityProportions = mutationProbabilityProportions;
    }

    @Override
    public double getSexualMutationProbability() {
        return sexualMutationProbability;
    }

    @Override
    public MutationSettings getConnectionWeightsMutationSettings() {
        return connectionWeightsMutationSettings;
    }

    @Override
    public Map<NeatStructuralMutationType, Double> getMutationProbabilityProportions() {
        return mutationProbabilityProportions;
    }
}
