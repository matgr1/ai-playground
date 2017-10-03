package matgr.ai.neuralnet.cyclic;

public enum ActivationResultCode {
    Success,
    PreSynapticNaN,
    PreSynapticInfinite,
    PostSynapticNaN,
    PostSynapticInfinite
}
