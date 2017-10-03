package matgr.ai.genetic;

public interface EvolutionParameters {

    double getEliteProportion();

    int getEliteCopies();

    double getAsexualReproductionProportion();

    double getSexualReproductionProportion();

    double getInterSpeciesSexualReproductionProportion();
}
