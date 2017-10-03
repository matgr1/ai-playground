package matgr.ai.math;

import org.apache.commons.math3.distribution.BetaDistribution;
import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.random.RandomGenerator;

import java.util.*;
import java.util.logging.Logger;

public final class RandomFunctions {

    private static Logger logger = Logger.getLogger(RandomFunctions.class.getName());

    public static double probabilisticRound(RandomGenerator random, double a) {
        double integerPart = Math.floor(a);
        double fractionalPart = a - integerPart;

        return (random.nextDouble() < fractionalPart) ? integerPart + 1.0 : integerPart;
    }

    public static boolean testProbability(RandomGenerator random, double probability) {
        return random.nextDouble() < probability;
    }

    public static double nextSign(RandomGenerator random) {
        if (random.nextDouble() < 0.5) {
            return -1.0;
        }

        return 1.0;
    }

    public static List<Double> doubleList(RandomGenerator random, int count, double minInclusive, double maxExclusive) {
        List<Double> values = new ArrayList<>();

        for (int i = 0; i < count; i++) {
            double value = nextDouble(random, minInclusive, maxExclusive);
            values.add(value);
        }

        return values;
    }

    public static double nextDouble(RandomGenerator random, double minInclusive, double maxExclusive) {
        if (minInclusive == maxExclusive) {
            return minInclusive;
        }

        double next = random.nextDouble();

        if ((0.0 == minInclusive) && (1.0 == maxExclusive)) {
            return next;
        }

        double range = maxExclusive - minInclusive;
        return minInclusive + (next * range);
    }

    public static <T> T selectItem(RandomGenerator random, Iterable<T> items) {
        return selectItem(random, items, null);
    }

    public static <T> T selectItem(RandomGenerator random, Iterable<T> items, T defaultValue) {
        if (null == items) {
            throw new IllegalArgumentException("items not provided");
        }

        Class<?> itemsType = items.getClass();

        if (List.class.isAssignableFrom(itemsType)) {

            List<T> lst = (List<T>) items;

            if (lst.size() < 1) {
                return defaultValue;
            }

            int index = random.nextInt(lst.size());
            return lst.get(index);
        }

        if (Collection.class.isAssignableFrom(itemsType)) {

            logger.warning("Falling back to less efficient selection algorithm");

            Collection<T> coll = (Collection<T>) items;

            if (coll.size() < 1) {
                return defaultValue;
            }

            int index = random.nextInt(coll.size());

            Optional<T> value = coll.stream().skip(index).findFirst();
            return value.orElse(null);
        }

        logger.warning("Falling back to less efficient selection algorithm");

        List<T> newList = new ArrayList<>();
        items.forEach(newList::add);

        return selectItem(random, newList, defaultValue);
    }

    public static int randomIndex(RandomGenerator random, int minIndex, int maxIndex) {

        int maxIndexShifted = maxIndex - minIndex;
        int nextShifted = random.nextInt(maxIndexShifted + 1);

        return nextShifted + minIndex;
    }

    public static double sampleNormalDistribution(RandomGenerator random, double mean, double standardDeviation) {
        NormalDistribution b = new NormalDistribution(random, mean, standardDeviation);
        return b.sample();
    }

    public static double sampleBetaDistribution(RandomGenerator random, double alpha, double beta) {
        BetaDistribution b = new BetaDistribution(random, alpha, beta);
        return b.sample();
    }

}
