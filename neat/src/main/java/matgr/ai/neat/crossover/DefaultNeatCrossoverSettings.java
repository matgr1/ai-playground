package matgr.ai.neat.crossover;

import matgr.ai.genetic.crossover.CrossoverSettings;

public class DefaultNeatCrossoverSettings implements NeatCrossoverSettings {

    public double probability;

    public double connectionCrossoverDisableRate;

    public CrossoverSettings crossoverSettings;

    public DefaultNeatCrossoverSettings(double probability,
                                        double connectionCrossoverDisableRate,
                                        CrossoverSettings crossoverSettings) {
        this.probability = probability;
        this.connectionCrossoverDisableRate = connectionCrossoverDisableRate;
        this.crossoverSettings = crossoverSettings;
    }

    @Override
    public double getProbability() {
        return probability;
    }

    @Override
    public double getConnectionCrossoverDisableRate() {
        return connectionCrossoverDisableRate;
    }

    @Override
    public CrossoverSettings getConnectionWeightsCrossoverSettings() {
        return crossoverSettings;
    }
}
