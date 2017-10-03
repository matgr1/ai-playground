package matgr.ai.genetic;

public interface SpeciesMember<GenomeT extends Genome> {

    GenomeT genome();

    double computeFitness();
}
