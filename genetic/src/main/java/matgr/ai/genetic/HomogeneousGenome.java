package matgr.ai.genetic;

import javax.annotation.Nonnull;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.UUID;

public class HomogeneousGenome<GeneT extends Number> implements Genome, Iterable<GeneT> {

    private final List<GeneT> genes;

    private final UUID genomeId;

    public HomogeneousGenome(Iterable<GeneT> genes) {
        this(UUID.randomUUID(), genes);
    }

    protected HomogeneousGenome(UUID genomeId, Iterable<GeneT> genes) {

        if (null == genomeId) {
            throw new IllegalArgumentException("genomeId not provided");
        }
        if (null == genes) {
            throw new IllegalArgumentException("genes not provided");
        }

        this.genomeId = genomeId;

        this.genes = new ArrayList<>();
        genes.forEach(this.genes::add);
    }

    public UUID genomeId() {
        return genomeId;
    }

    public int geneCount() {
        return genes.size();
    }

    public GeneT getGene(int index) {
        return genes.get(index);
    }

    protected void setGene(int index, GeneT value) {
        genes.set(index, value);
    }

    @Override
    public HomogeneousGenome<GeneT> deepClone(UUID genomeId) {
        return new HomogeneousGenome<>(genomeId, this);
    }

    @Override
    @Nonnull
    public Iterator<GeneT> iterator() {
        return genes.iterator();
    }
}
