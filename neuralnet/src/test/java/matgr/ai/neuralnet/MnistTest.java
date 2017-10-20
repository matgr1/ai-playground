package matgr.ai.neuralnet;

import junit.framework.Test;
import junit.framework.TestCase;
import junit.framework.TestSuite;
import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;

import java.io.DataInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;


/**
 * Unit test for simple App.
 */
public class MnistTest extends TestCase {


    private static final RandomGenerator random = new MersenneTwister();

    /**
     * Create the test case
     *
     * @param testName name of the test case
     */
    public MnistTest(String testName) {

        super(testName);
    }

    /**
     * @return the suite of tests being tested
     */
    public static Test suite() {
        return new TestSuite(NeuralNetTest.class);
    }

    public void testMnist() throws IOException {

        MnistIdxFile trainingLabels = loadMnistIdxResourceFile("/train-labels.idx1-ubyte");
        MnistIdxFile trainingImages = loadMnistIdxResourceFile("/train-images.idx3-ubyte");
    }

    private MnistIdxFile loadMnistIdxResourceFile(String path) throws IOException {

        try (InputStream stream = getClass().getResourceAsStream(path)) {
            return loadMnistIdxResourceFile(stream);
        }
    }

    private MnistIdxFile loadMnistIdxResourceFile(InputStream stream) throws IOException {

        // NOTE: this is big endian I guess... doesn't really say here (which would be nice):
        //       https://docs.oracle.com/javase/8/docs/api/java/io/DataInputStream.html, but there is a claim
        //       here that it is: https://stackoverflow.com/questions/13211770/endianness-on-datainputstream
        try (DataInputStream bigEndianStream = new DataInputStream(stream)) {

            // TODO: do this better, format is here: http://yann.lecun.com/exdb/mnist/

            int magicNumber = bigEndianStream.readInt();

            int magic0 = (magicNumber >>> 24) & 0xff;
            int magic1 = (magicNumber >>> 16) & 0xff;

            int dataType = (magicNumber >>> 8) & 0xff;
            int dimensionCount = magicNumber & 0xff;

            if ((magic0 != 0) || (magic1 != 0)) {
                throw new IllegalArgumentException("Invalid MNIST IDX file");
            }

            if (dataType != 0x08) {
                throw new IllegalArgumentException("Only single byte data supported for now...");
            }

            int itemCount;
            int itemWidth;
            int itemHeight;

            switch (dimensionCount) {
                case 1: {
                    itemCount = bigEndianStream.readInt();
                    itemHeight = 1;
                    itemWidth = 1;
                }
                break;

                case 3: {
                    itemCount = bigEndianStream.readInt();
                    itemHeight = bigEndianStream.readInt();
                    itemWidth = bigEndianStream.readInt();
                }
                break;

                default:
                    throw new IllegalArgumentException("Only dimensions counts of 1 and 3 are supported");
            }

            List<byte[]> data = new ArrayList<>();

            int itemSize = itemHeight * itemWidth;

            for (int i = 0; i < itemCount; i++) {

                byte[] curItem = new byte[itemSize];

                if (bigEndianStream.read(curItem) != itemSize) {
                    throw new IllegalArgumentException("Failed to read item");
                }

                data.add(curItem);
            }

            return new MnistIdxFile(itemWidth, itemHeight, data);
        }
    }

    private static class MnistIdxFile {

        public final int itemWidth;
        public final int itemHeight;

        public final List<byte[]> data;

        private MnistIdxFile(int itemWidth, int itemHeight, List<byte[]> data) {
            this.itemWidth = itemWidth;
            this.itemHeight = itemHeight;
            this.data = data;
        }
    }
}
