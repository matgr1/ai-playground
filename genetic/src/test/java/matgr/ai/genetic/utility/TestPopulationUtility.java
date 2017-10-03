package matgr.ai.genetic.utility;

import matgr.ai.genetic.NumericGenome;
import matgr.ai.math.RandomFunctions;
import org.apache.commons.math3.random.RandomGenerator;

import java.util.ArrayList;
import java.util.List;

public class TestPopulationUtility {

    public static List<NumericGenome> createRandomGenomes(RandomGenerator random, int populationSize, int genomeLength) {

        List<NumericGenome> genomes = new ArrayList<>();

        for (int n = 0; n < populationSize; n++) {
            List<Double> genes = new ArrayList<>();

            for (int i = 0; i < genomeLength; i++) {
                double gene = RandomFunctions.nextDouble(random, -1.0, 1.0);
                genes.add(gene);
            }

            NumericGenome genome = new NumericGenome(genes);
            genomes.add(genome);
        }

        return genomes;

    }

}
