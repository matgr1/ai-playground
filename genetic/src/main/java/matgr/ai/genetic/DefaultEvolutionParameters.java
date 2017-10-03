package matgr.ai.genetic;

public class DefaultEvolutionParameters implements EvolutionParameters {

    public double eliteProportion;
    public int eliteCopies;
    public double asexualReproductionProportion;
    public double sexualReproductionProportion;
    public double interSpeciesSexualReproductionProportion;

    public DefaultEvolutionParameters(double eliteProportion,
                                      int eliteCopies,
                                      double asexualReproductionProportion,
                                      double sexualReproductionProportion,
                                      double interSpeciesSexualReproductionProportion) {
        this.eliteProportion = eliteProportion;
        this.eliteCopies = eliteCopies;
        this.asexualReproductionProportion = asexualReproductionProportion;
        this.sexualReproductionProportion = sexualReproductionProportion;
        this.interSpeciesSexualReproductionProportion = interSpeciesSexualReproductionProportion;
    }

    @Override
    public double getEliteProportion() {
        return 0;
    }

    @Override
    public int getEliteCopies() {
        return 0;
    }

    @Override
    public double getAsexualReproductionProportion() {
        return 0;
    }

    @Override
    public double getSexualReproductionProportion() {
        return 0;
    }

    @Override
    public double getInterSpeciesSexualReproductionProportion() {
        return 0;
    }
}
