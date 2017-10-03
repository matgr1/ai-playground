package matgr.ai.genetic.crossover;

import matgr.ai.genetic.ValueRangeSettings;

public interface CrossoverSettings extends ValueRangeSettings {

    CrossoverType getNumericCrossoverType();
}
