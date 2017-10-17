//package matgr.ai.neuralnet.feedforward;
//
//import matgr.ai.common.SizedIterable;
//import matgr.ai.neuralnet.Neuron;
//import matgr.ai.neuralnet.NeuronState;
//import org.apache.commons.collections4.iterators.ArrayIterator;
//import sun.reflect.generics.reflectiveObjects.NotImplementedException;
//
//public class SoftmaxLayerActivationFunction implements LayerActivationFunction {
//
//    @Override
//    public <NeuronT extends Neuron> void activate(SizedIterable<NeuronState<NeuronT>> neurons) {
//
//        // NOTE: this helps mitigate numerical stability issues... (see here:
//        //       https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/)
//
//        double maxPreSynapse = Double.NEGATIVE_INFINITY;
//
//        for (NeuronState<NeuronT> neuron : neurons) {
//            maxPreSynapse = Math.max(maxPreSynapse, neuron.preSynapse);
//        }
//
//        double d = -maxPreSynapse;
//
//        // compute the overall sum (and save the intermediate values)
//
//        double expSum = 0.0;
//        double[] exps = new double[neurons.size()];
//
//        int index = 0;
//
//        for (NeuronState<NeuronT> neuron : neurons) {
//
//            double curExp = Math.exp(neuron.preSynapse + d);
//
//            exps[index++] = curExp;
//            expSum += curExp;
//        }
//
//        // compute the activations
//
//        ArrayIterator<Double> expsIterator = new ArrayIterator<>(exps);
//
//        for (NeuronState<NeuronT> neuron : neurons) {
//
//            double curExp = expsIterator.next();
//            neuron.postSynapse = curExp / expSum;
//        }
//    }
//
//    @Override
//    public double computeDerivative(double activationOutput) {
//        throw new NotImplementedException();
//    }
//}
