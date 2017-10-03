package matgr.ai.genetic;

public class GeneticAlgorithmResult<SpeciesMemberT extends SpeciesMember> {

    public final FitnessItem<SpeciesMemberT> bestMatch;

    public final int iterations;

    public final boolean success;

    public GeneticAlgorithmResult(FitnessItem<SpeciesMemberT> bestMatch,
                                  int iterations,
                                  boolean success) {

        this.bestMatch = bestMatch;
        this.iterations = iterations;
        this.success = success;

    }

}
