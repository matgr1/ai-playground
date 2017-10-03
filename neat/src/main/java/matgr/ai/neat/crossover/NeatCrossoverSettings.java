package matgr.ai.neat.crossover;

import matgr.ai.genetic.crossover.CrossoverSettings;

public interface NeatCrossoverSettings {

    double getProbability();

    double getConnectionCrossoverDisableRate();

    CrossoverSettings getConnectionWeightsCrossoverSettings();
}
