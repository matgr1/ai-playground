package matgr.ai.genetic;

import matgr.ai.math.MathFunctions;
import matgr.ai.math.RandomFunctions;
import org.apache.commons.math3.random.RandomGenerator;

public interface ValueRangeSettings {

    double getValueRange();

    default double clampValueToRange(double value) {
        double valueRange = getValueRange();
        return MathFunctions.clamp(value, -valueRange, valueRange);
    }

    default double getRandomValueInRange(RandomGenerator random) {
        double valueRange = getValueRange();
        return getRandomValueInRange(random, valueRange);
    }

    static double getRandomValueInRange(RandomGenerator random, double valueRange) {
        return RandomFunctions.nextDouble(random, -valueRange, valueRange);
    }

}
