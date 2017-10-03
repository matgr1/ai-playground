package matgr.ai.genetic;

public class SortedGenomeParents<T> {

    public FitnessItem<T> fittest;

    public FitnessItem<T> other;

    public SortedGenomeParents(FitnessItem<T> fittest, FitnessItem<T> other) {
        this.fittest = fittest;
        this.other = other;
    }

}
