package matgr.ai.genetic;

import matgr.ai.math.RandomFunctions;
import org.apache.commons.math3.random.RandomGenerator;

import java.util.ArrayList;
import java.util.List;

public class GenomeParents<T> {

    private final SortedGenomeParents<T> sorted;
    private final List<FitnessItem<T>> unsorted;

    public GenomeParents(FitnessItem<T> item1, FitnessItem<T> item2) throws IllegalArgumentException {

        if ((null == item1) && (null == item2)) {
            throw new IllegalArgumentException("Must provide at least one item");
        }

        if (null == item1) {

            if (Double.isNaN(item2.fitness)) {
                throw new IllegalArgumentException("Invalid fitness item");
            }

            sorted = new SortedGenomeParents<>(item2, null);
            unsorted = null;

        } else if (null == item2) {

            if (Double.isNaN(item1.fitness)) {
                throw new IllegalArgumentException("Invalid fitness item");
            }

            sorted = new SortedGenomeParents<>(item1, null);
            unsorted = null;

        } else {

            if (Double.isNaN(item1.fitness) || Double.isNaN(item2.fitness)) {
                throw new IllegalArgumentException("Invalid fitness item");
            }

            if (item1.fitness > item2.fitness) {

                sorted = new SortedGenomeParents<>(item1, item2);
                unsorted = null;

            } else if (item1.fitness < item2.fitness) {

                sorted = new SortedGenomeParents<>(item2, item1);
                unsorted = null;

            } else {

                sorted = null;
                unsorted = new ArrayList<>();
                unsorted.add(item1);
                unsorted.add(item2);

            }
        }
    }

    public SortedGenomeParents<T> getSorted(RandomGenerator random) {

        if (null != sorted) {
            return sorted;
        }

        // NOTE: this makes sure that if there is no clear winner, each call has a chance of sorting differently
        // TODO: it's a little weird to handle it like this, it should really be up to the caller to handle this case

        int fittest = RandomFunctions.randomIndex(random, 0, 1);
        int other = 1 - fittest;

        return new SortedGenomeParents<>(unsorted.get(fittest), unsorted.get(other));

    }
}