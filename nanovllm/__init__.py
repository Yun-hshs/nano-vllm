from nanovllm.sampling_params import SamplingParams

__all__ = ["LLM", "SamplingParams"]


def __getattr__(name):
    # `LLM` drags in torch, torch.distributed and the attention kernels. Import
    # it lazily so host-side pieces (the scheduler, block manager, sampling
    # params) can be used and unit-tested without the CUDA model stack.
    if name == "LLM":
        from nanovllm.llm import LLM
        return LLM
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
