package matgr.ai.genetic;

import java.util.UUID;

public class NumericGenome extends HomogeneousGenome<Double> {

    public NumericGenome(Iterable<Double> genes) {
        super(genes);
    }

    public NumericGenome(UUID genomeId, Iterable<Double> genes) {
        super(genomeId, genes);
    }

    @Override
    public void setGene(int index, Double value)
    {
        super.setGene(index, value);
    }

    @Override
    public NumericGenome deepClone(UUID genomeId) {
        return new NumericGenome(genomeId, this);
    }

}
