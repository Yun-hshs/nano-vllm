from __future__ import annotations

from collections.abc import Callable, Sequence as SequenceLike
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from time import perf_counter


@dataclass(frozen=True, slots=True)
class DSparkSpeculativeConfig:
    target_model: str = "Qwen/Qwen3-4B"
    draft_model: str = "deepseek-ai/dspark_qwen3_4b_block7"
    num_speculative_tokens: int = 7
    target_layer_ids: tuple[int, ...] = (1, 9, 17, 25, 33)
    num_anchors: int = 512
    markov_rank: int = 256
    confidence_threshold: float = 0.0
    confidence_head_with_markov: bool = True

    def __post_init__(self):
        if self.num_speculative_tokens <= 0:
            raise ValueError("num_speculative_tokens must be positive")
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class SpeculativeRuntimeConfig:
    enabled: bool = False
    draft_type: str = "ngram"
    draft_model: str = "Qwen/Qwen3-0.6B"
    draft_device: str = "cuda"
    draft_dtype: str = "auto"
    num_speculative_tokens: int = 7
    ngram_size: int = 4
    confidence_threshold: float = 0.0
    greedy_verification_temperature: float = 0.011

    def __post_init__(self):
        if self.num_speculative_tokens <= 0:
            raise ValueError("num_speculative_tokens must be positive")
        if self.ngram_size <= 0:
            raise ValueError("ngram_size must be positive")
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")
        if self.greedy_verification_temperature < 0.0:
            raise ValueError("greedy_verification_temperature must be non-negative")


def should_use_greedy_verification(temperature: float, threshold: float = 0.011) -> bool:
    return threshold > 0.0 and temperature <= threshold


@dataclass(frozen=True, slots=True)
class DraftProposal:
    token_ids: list[int]
    confidence: list[float] | None = None

    def __post_init__(self):
        if self.confidence is not None and len(self.confidence) != len(self.token_ids):
            raise ValueError("confidence length must match token_ids length")


@dataclass(frozen=True, slots=True)
class VerificationResult:
    accepted_token_ids: list[int]
    rejected_token_id: int | None
    proposed_tokens: int
    accepted_tokens: int


@dataclass(frozen=True, slots=True)
class TargetOnlyExperimentResult:
    output_token_ids: list[int]
    proposed_tokens: int
    accepted_tokens: int
    target_forwards: int
    elapsed_seconds: float
    acceptance_lengths: list[int]

    @property
    def tokens_per_second(self) -> float:
        if self.elapsed_seconds <= 0:
            raise ValueError("elapsed_seconds must be positive")
        return len(self.output_token_ids) / self.elapsed_seconds

    @property
    def mean_acceptance_length(self) -> float:
        if not self.acceptance_lengths:
            return 0.0
        return sum(self.acceptance_lengths) / len(self.acceptance_lengths)


@dataclass(slots=True)
class SpeculativeStats:
    proposed_tokens: int = 0
    accepted_tokens: int = 0
    draft_forwards: int = 0
    draft_rebuilds: int = 0
    draft_cached_steps: int = 0
    draft_graph_enabled: bool = False
    draft_graph_replays: int = 0
    target_forwards: int = 0
    target_greedy_forwards: int = 0
    target_greedy_tokens: int = 0
    emitted_tokens: int = 0
    draft_propose_ms: float = 0.0
    draft_prefill_ms: float = 0.0
    draft_cache_extend_ms: float = 0.0
    draft_decode_graph_ms: float = 0.0
    draft_decode_eager_ms: float = 0.0
    draft_token_select_ms: float = 0.0
    target_verify_ms: float = 0.0
    target_forward_ms: float = 0.0
    target_argmax_ms: float = 0.0
    target_compare_ms: float = 0.0
    append_tokens_ms: float = 0.0
    speculative_step_ms: float = 0.0
    acceptance_lengths: list[int] | None = None

    def __post_init__(self):
        if self.acceptance_lengths is None:
            self.acceptance_lengths = []

    @property
    def acceptance_rate(self) -> float | None:
        if self.proposed_tokens == 0:
            return None
        return self.accepted_tokens / self.proposed_tokens

    @property
    def mean_acceptance_length(self) -> float:
        if not self.acceptance_lengths:
            return 0.0
        return sum(self.acceptance_lengths) / len(self.acceptance_lengths)

    def record_step(
        self,
        *,
        proposed_tokens: int,
        accepted_tokens: int,
        target_forwards: int,
        emitted_tokens: int,
        draft_forwards: int = 0,
        draft_rebuilds: int = 0,
        draft_cached_steps: int = 0,
        draft_graph_enabled: bool = False,
        draft_graph_replays: int = 0,
        target_greedy_forwards: int = 0,
        target_greedy_tokens: int = 0,
        draft_propose_ms: float = 0.0,
        draft_prefill_ms: float = 0.0,
        draft_cache_extend_ms: float = 0.0,
        draft_decode_graph_ms: float = 0.0,
        draft_decode_eager_ms: float = 0.0,
        draft_token_select_ms: float = 0.0,
        target_verify_ms: float = 0.0,
        target_forward_ms: float = 0.0,
        target_argmax_ms: float = 0.0,
        target_compare_ms: float = 0.0,
        append_tokens_ms: float = 0.0,
        speculative_step_ms: float = 0.0,
    ):
        self.proposed_tokens += proposed_tokens
        self.accepted_tokens += accepted_tokens
        self.draft_forwards += draft_forwards
        self.draft_rebuilds += draft_rebuilds
        self.draft_cached_steps += draft_cached_steps
        self.draft_graph_enabled = self.draft_graph_enabled or draft_graph_enabled
        self.draft_graph_replays += draft_graph_replays
        self.target_forwards += target_forwards
        self.target_greedy_forwards += target_greedy_forwards
        self.target_greedy_tokens += target_greedy_tokens
        self.emitted_tokens += emitted_tokens
        self.draft_propose_ms += draft_propose_ms
        self.draft_prefill_ms += draft_prefill_ms
        self.draft_cache_extend_ms += draft_cache_extend_ms
        self.draft_decode_graph_ms += draft_decode_graph_ms
        self.draft_decode_eager_ms += draft_decode_eager_ms
        self.draft_token_select_ms += draft_token_select_ms
        self.target_verify_ms += target_verify_ms
        self.target_forward_ms += target_forward_ms
        self.target_argmax_ms += target_argmax_ms
        self.target_compare_ms += target_compare_ms
        self.append_tokens_ms += append_tokens_ms
        self.speculative_step_ms += speculative_step_ms
        self.acceptance_lengths.append(accepted_tokens)

    def reset(self):
        self.proposed_tokens = 0
        self.accepted_tokens = 0
        self.draft_forwards = 0
        self.draft_rebuilds = 0
        self.draft_cached_steps = 0
        self.draft_graph_enabled = False
        self.draft_graph_replays = 0
        self.target_forwards = 0
        self.target_greedy_forwards = 0
        self.target_greedy_tokens = 0
        self.emitted_tokens = 0
        self.draft_propose_ms = 0.0
        self.draft_prefill_ms = 0.0
        self.draft_cache_extend_ms = 0.0
        self.draft_decode_graph_ms = 0.0
        self.draft_decode_eager_ms = 0.0
        self.draft_token_select_ms = 0.0
        self.target_verify_ms = 0.0
        self.target_forward_ms = 0.0
        self.target_argmax_ms = 0.0
        self.target_compare_ms = 0.0
        self.append_tokens_ms = 0.0
        self.speculative_step_ms = 0.0
        self.acceptance_lengths.clear()


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    mode: str
    output_tokens: int
    elapsed_seconds: float
    proposed_tokens: int = 0
    accepted_tokens: int = 0
    draft_forwards: int = 0
    draft_rebuilds: int = 0
    draft_cached_steps: int = 0
    draft_graph_enabled: bool = False
    draft_graph_replays: int = 0
    target_forwards: int = 0
    target_greedy_forwards: int = 0
    target_greedy_tokens: int = 0
    draft_propose_ms: float = 0.0
    draft_prefill_ms: float = 0.0
    draft_cache_extend_ms: float = 0.0
    draft_decode_graph_ms: float = 0.0
    draft_decode_eager_ms: float = 0.0
    draft_token_select_ms: float = 0.0
    target_verify_ms: float = 0.0
    target_forward_ms: float = 0.0
    target_argmax_ms: float = 0.0
    target_compare_ms: float = 0.0
    append_tokens_ms: float = 0.0
    speculative_step_ms: float = 0.0
    acceptance_lengths: list[int] = field(default_factory=list)

    @property
    def tokens_per_second(self) -> float:
        if self.elapsed_seconds <= 0:
            raise ValueError("elapsed_seconds must be positive")
        return self.output_tokens / self.elapsed_seconds

    @property
    def acceptance_rate(self) -> float | None:
        if self.proposed_tokens == 0:
            return None
        return self.accepted_tokens / self.proposed_tokens

    @property
    def mean_acceptance_length(self) -> float:
        if not self.acceptance_lengths:
            return 0.0
        return sum(self.acceptance_lengths) / len(self.acceptance_lengths)

    def to_dict(self) -> dict[str, int | float | str]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class BenchmarkComparison:
    baseline_tokens_per_second: float
    speculative_tokens_per_second: float
    speedup: float
    acceptance_rate: float | None


DraftFn = Callable[[list[int], int], DraftProposal]
TargetFn = Callable[[list[int], int], SequenceLike[int]]


def verify_draft_tokens(
    proposal: DraftProposal,
    target_token_ids: SequenceLike[int],
    confidence_threshold: float = 0.0,
) -> VerificationResult:
    accepted: list[int] = []
    rejected_token_id: int | None = None

    for idx, token_id in enumerate(proposal.token_ids):
        if proposal.confidence is not None and proposal.confidence[idx] < confidence_threshold:
            rejected_token_id = token_id
            break
        if idx >= len(target_token_ids) or token_id != target_token_ids[idx]:
            rejected_token_id = token_id
            break
        accepted.append(token_id)

    return VerificationResult(
        accepted_token_ids=accepted,
        rejected_token_id=rejected_token_id,
        proposed_tokens=len(proposal.token_ids),
        accepted_tokens=len(accepted),
    )


def compare_benchmarks(
    baseline: BenchmarkResult,
    speculative: BenchmarkResult,
) -> BenchmarkComparison:
    baseline_tps = baseline.tokens_per_second
    speculative_tps = speculative.tokens_per_second
    return BenchmarkComparison(
        baseline_tokens_per_second=baseline_tps,
        speculative_tokens_per_second=speculative_tps,
        speedup=speculative_tps / baseline_tps,
        acceptance_rate=speculative.acceptance_rate,
    )


def benchmark_result_from_dict(data: dict) -> BenchmarkResult:
    return BenchmarkResult(
        mode=str(data["mode"]),
        output_tokens=int(data["output_tokens"]),
        elapsed_seconds=float(data["elapsed_seconds"]),
        proposed_tokens=int(data.get("proposed_tokens", 0)),
        accepted_tokens=int(data.get("accepted_tokens", 0)),
        draft_forwards=int(data.get("draft_forwards", 0)),
        draft_rebuilds=int(data.get("draft_rebuilds", 0)),
        draft_cached_steps=int(data.get("draft_cached_steps", 0)),
        draft_graph_enabled=bool(data.get("draft_graph_enabled", False)),
        draft_graph_replays=int(data.get("draft_graph_replays", 0)),
        target_forwards=int(data.get("target_forwards", 0)),
        target_greedy_forwards=int(data.get("target_greedy_forwards", 0)),
        target_greedy_tokens=int(data.get("target_greedy_tokens", 0)),
        draft_propose_ms=float(data.get("draft_propose_ms", 0.0)),
        draft_prefill_ms=float(data.get("draft_prefill_ms", 0.0)),
        draft_cache_extend_ms=float(data.get("draft_cache_extend_ms", 0.0)),
        draft_decode_graph_ms=float(data.get("draft_decode_graph_ms", 0.0)),
        draft_decode_eager_ms=float(data.get("draft_decode_eager_ms", 0.0)),
        draft_token_select_ms=float(data.get("draft_token_select_ms", 0.0)),
        target_verify_ms=float(data.get("target_verify_ms", 0.0)),
        target_forward_ms=float(data.get("target_forward_ms", 0.0)),
        target_argmax_ms=float(data.get("target_argmax_ms", 0.0)),
        target_compare_ms=float(data.get("target_compare_ms", 0.0)),
        append_tokens_ms=float(data.get("append_tokens_ms", 0.0)),
        speculative_step_ms=float(data.get("speculative_step_ms", 0.0)),
        acceptance_lengths=[int(value) for value in data.get("acceptance_lengths", [])],
    )


def format_benchmark_comparison(
    baseline: BenchmarkResult,
    speculative: BenchmarkResult,
) -> str:
    comparison = compare_benchmarks(baseline, speculative)
    acceptance = speculative.acceptance_rate
    baseline_acceptance = "-"
    speculative_acceptance = "-" if acceptance is None else f"{acceptance * 100:.2f}%"
    return "\n".join([
        "# Benchmark Comparison",
        "",
        "| mode | output tokens | time (s) | throughput (tok/s) | acceptance | mean accepted | draft forwards | draft rebuilds | cached steps | graph enabled | graph replays | target forwards | greedy forwards | draft propose ms | draft prefill ms | draft cache extend ms | draft graph ms | draft eager ms | draft select ms | target verify ms | target forward ms | target argmax ms | target compare ms | append ms | step ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| {baseline.mode} | {baseline.output_tokens} | "
            f"{baseline.elapsed_seconds:.2f} | {baseline.tokens_per_second:.2f} | "
            f"{baseline_acceptance} | {baseline.mean_acceptance_length:.2f} | "
            f"{baseline.draft_forwards} | "
            f"{baseline.draft_rebuilds} | "
            f"{baseline.draft_cached_steps} | "
            f"{'yes' if baseline.draft_graph_enabled else 'no'} | "
            f"{baseline.draft_graph_replays} | "
            f"{baseline.target_forwards} | "
            f"{baseline.target_greedy_forwards} | "
            f"{baseline.draft_propose_ms:.2f} | "
            f"{baseline.draft_prefill_ms:.2f} | "
            f"{baseline.draft_cache_extend_ms:.2f} | "
            f"{baseline.draft_decode_graph_ms:.2f} | "
            f"{baseline.draft_decode_eager_ms:.2f} | "
            f"{baseline.draft_token_select_ms:.2f} | "
            f"{baseline.target_verify_ms:.2f} | "
            f"{baseline.target_forward_ms:.2f} | "
            f"{baseline.target_argmax_ms:.2f} | "
            f"{baseline.target_compare_ms:.2f} | "
            f"{baseline.append_tokens_ms:.2f} | "
            f"{baseline.speculative_step_ms:.2f} |"
        ),
        (
            f"| {speculative.mode} | {speculative.output_tokens} | "
            f"{speculative.elapsed_seconds:.2f} | {speculative.tokens_per_second:.2f} | "
            f"{speculative_acceptance} | {speculative.mean_acceptance_length:.2f} | "
            f"{speculative.draft_forwards} | "
            f"{speculative.draft_rebuilds} | "
            f"{speculative.draft_cached_steps} | "
            f"{'yes' if speculative.draft_graph_enabled else 'no'} | "
            f"{speculative.draft_graph_replays} | "
            f"{speculative.target_forwards} | "
            f"{speculative.target_greedy_forwards} | "
            f"{speculative.draft_propose_ms:.2f} | "
            f"{speculative.draft_prefill_ms:.2f} | "
            f"{speculative.draft_cache_extend_ms:.2f} | "
            f"{speculative.draft_decode_graph_ms:.2f} | "
            f"{speculative.draft_decode_eager_ms:.2f} | "
            f"{speculative.draft_token_select_ms:.2f} | "
            f"{speculative.target_verify_ms:.2f} | "
            f"{speculative.target_forward_ms:.2f} | "
            f"{speculative.target_argmax_ms:.2f} | "
            f"{speculative.target_compare_ms:.2f} | "
            f"{speculative.append_tokens_ms:.2f} | "
            f"{speculative.speculative_step_ms:.2f} |"
        ),
        "",
        f"Speedup: {comparison.speedup:.2f}x",
    ])


class SpeculativeDecoder:
    """Small DSpark-style proposal/verify loop with pluggable model callbacks."""

    def __init__(
        self,
        draft_fn: DraftFn,
        target_fn: TargetFn,
        config: DSparkSpeculativeConfig | None = None,
    ):
        self.config = config or DSparkSpeculativeConfig()
        self.draft_fn = draft_fn
        self.target_fn = target_fn

    def step(self, prefix_token_ids: list[int]) -> VerificationResult:
        proposal = self.draft_fn(prefix_token_ids, self.config.num_speculative_tokens)
        target_token_ids = self.target_fn(prefix_token_ids, len(proposal.token_ids))
        return verify_draft_tokens(
            proposal,
            target_token_ids,
            confidence_threshold=self.config.confidence_threshold,
        )


class NGramDraftGenerator:
    """Drafts by replaying tokens that followed the latest matching n-gram."""

    def __init__(self, ngram_size: int = 4):
        if ngram_size <= 0:
            raise ValueError("ngram_size must be positive")
        self.ngram_size = ngram_size

    def propose(self, prefix_token_ids: list[int], num_tokens: int) -> DraftProposal:
        if num_tokens <= 0:
            return DraftProposal([])
        if len(prefix_token_ids) < self.ngram_size:
            return DraftProposal([])

        key = prefix_token_ids[-self.ngram_size:]
        max_proposal_end = len(prefix_token_ids) - self.ngram_size
        for start in range(len(prefix_token_ids) - self.ngram_size - 1, -1, -1):
            if prefix_token_ids[start:start + self.ngram_size] == key:
                proposal_start = start + self.ngram_size
                proposal_end = min(proposal_start + num_tokens, max_proposal_end)
                proposal = prefix_token_ids[proposal_start:proposal_end]
                return DraftProposal(proposal, [1.0] * len(proposal))
        return DraftProposal([])


class HFDraftGenerator:
    """Greedy Hugging Face draft model wrapper for token-id proposals."""

    def __init__(
        self,
        model_name_or_path: str,
        *,
        device: str = "cuda",
        dtype: str = "auto",
    ):
        import torch
        from transformers import AutoModelForCausalLM

        torch_dtype = dtype
        if dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        elif dtype == "float16":
            torch_dtype = torch.float16
        elif dtype == "float32":
            torch_dtype = torch.float32
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            dtype=torch_dtype,
        )
        if device != "auto":
            self.model = self.model.to(device)
        self.model = self.model.eval()
        self._cached_prefix_token_ids: list[int] = []
        self._past_key_values = None
        self._next_logits = None
        self.last_draft_forwards = 0
        self.last_draft_rebuilds = 0
        self.last_draft_cached_steps = 0

    @classmethod
    def from_model(cls, model):
        generator = cls.__new__(cls)
        generator.model = model.eval()
        generator.device = None
        generator._cached_prefix_token_ids = []
        generator._past_key_values = None
        generator._next_logits = None
        generator.last_draft_forwards = 0
        generator.last_draft_rebuilds = 0
        generator.last_draft_cached_steps = 0
        return generator

    def propose(self, prefix_token_ids: list[int], num_tokens: int) -> DraftProposal:
        if num_tokens <= 0:
            return DraftProposal([])
        self._reset_last_stats()
        try:
            import torch
        except ModuleNotFoundError:
            torch = None

        if torch is None or not callable(self.model):
            return self._generate_without_cache(prefix_token_ids, num_tokens, torch)

        device = self._get_device(torch)
        with torch.inference_mode():
            logits = self._ensure_cache(prefix_token_ids, device, torch)
            proposed_token_ids: list[int] = []
            for _ in range(num_tokens):
                token_id = int(torch.argmax(logits[:, -1, :], dim=-1).item())
                proposed_token_ids.append(token_id)
                logits = self._extend_cache([token_id], device, torch)
        return DraftProposal(proposed_token_ids, [1.0] * len(proposed_token_ids))

    def _reset_last_stats(self):
        self.last_draft_forwards = 0
        self.last_draft_rebuilds = 0
        self.last_draft_cached_steps = 0

    def _get_device(self, torch):
        if self.device is not None and self.device != "auto":
            return self.device
        try:
            return next(self.model.parameters()).device
        except (AttributeError, StopIteration):
            return "cpu"

    def _generate_without_cache(self, prefix_token_ids: list[int], num_tokens: int, torch) -> DraftProposal:
        if torch is None:
            input_ids = type("InputIds", (), {"shape": (1, len(prefix_token_ids))})()
            attention_mask = None
            inference_mode = nullcontext
        else:
            input_ids = torch.tensor([prefix_token_ids], dtype=torch.long, device=self._get_device(torch))
            attention_mask = torch.ones_like(input_ids)
            inference_mode = torch.inference_mode
        pad_token_id = getattr(self.model.generation_config, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(self.model.generation_config, "eos_token_id", None)
        if isinstance(pad_token_id, list):
            pad_token_id = pad_token_id[0] if pad_token_id else None
        self.last_draft_forwards = 1
        self.last_draft_rebuilds = 1
        with inference_mode():
            generated = self.model.generate(
                input_ids,
                max_new_tokens=num_tokens,
                do_sample=False,
                pad_token_id=pad_token_id,
                attention_mask=attention_mask,
            )
        token_ids = generated[0, input_ids.shape[1]:].tolist()
        return DraftProposal(token_ids, [1.0] * len(token_ids))

    def _ensure_cache(self, prefix_token_ids: list[int], device, torch):
        prefix = list(prefix_token_ids)
        if self._next_logits is not None and self._cached_prefix_token_ids == prefix:
            self.last_draft_cached_steps += len(prefix)
            return self._next_logits
        if (
            self._next_logits is not None
            and self._cached_prefix_token_ids
            and prefix[:len(self._cached_prefix_token_ids)] == self._cached_prefix_token_ids
        ):
            cached_len = len(self._cached_prefix_token_ids)
            self.last_draft_cached_steps += cached_len
            suffix = prefix[cached_len:]
            if suffix:
                return self._extend_cache(suffix, device, torch)
            return self._next_logits
        return self._rebuild_cache(prefix, device, torch)

    def _rebuild_cache(self, prefix_token_ids: list[int], device, torch):
        input_ids = torch.tensor([prefix_token_ids], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
        )
        self.last_draft_forwards += 1
        self.last_draft_rebuilds += 1
        self._past_key_values = outputs.past_key_values
        self._next_logits = outputs.logits
        self._cached_prefix_token_ids = list(prefix_token_ids)
        return self._next_logits

    def _extend_cache(self, token_ids: list[int], device, torch):
        if not token_ids:
            return self._next_logits
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
        attention_mask = torch.ones(
            (1, len(self._cached_prefix_token_ids) + len(token_ids)),
            dtype=torch.long,
            device=device,
        )
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=self._past_key_values,
            use_cache=True,
        )
        self.last_draft_forwards += 1
        self._past_key_values = outputs.past_key_values
        self._next_logits = outputs.logits
        self._cached_prefix_token_ids.extend(token_ids)
        return self._next_logits


def run_target_only_speculative_experiment(
    *,
    prefix_token_ids: list[int],
    max_tokens: int,
    draft_fn: DraftFn,
    target_fn: TargetFn,
    config: DSparkSpeculativeConfig | None = None,
    elapsed_seconds: float | None = None,
) -> TargetOnlyExperimentResult:
    config = config or DSparkSpeculativeConfig()
    prefix = list(prefix_token_ids)
    output: list[int] = []
    proposed_tokens = 0
    accepted_tokens = 0
    target_forwards = 0
    acceptance_lengths: list[int] = []
    start_time = perf_counter()

    while len(output) < max_tokens:
        remaining = max_tokens - len(output)
        proposal = draft_fn(prefix, min(config.num_speculative_tokens, remaining))
        if not proposal.token_ids:
            target_tokens = list(target_fn(prefix, 1))
            target_forwards += 1
            if not target_tokens:
                break
            next_token = target_tokens[0]
            prefix.append(next_token)
            output.append(next_token)
            acceptance_lengths.append(0)
            continue

        target_tokens = list(target_fn(prefix, len(proposal.token_ids)))
        target_forwards += 1
        verification = verify_draft_tokens(
            proposal,
            target_tokens,
            confidence_threshold=config.confidence_threshold,
        )
        proposed_tokens += verification.proposed_tokens
        accepted_tokens += verification.accepted_tokens
        acceptance_lengths.append(verification.accepted_tokens)

        for token_id in verification.accepted_token_ids[:remaining]:
            prefix.append(token_id)
            output.append(token_id)
        if len(output) >= max_tokens:
            break

        if len(target_tokens) > verification.accepted_tokens:
            next_token = target_tokens[verification.accepted_tokens]
            prefix.append(next_token)
            output.append(next_token)
        elif not target_tokens:
            break

    measured_elapsed = perf_counter() - start_time if elapsed_seconds is None else elapsed_seconds
    return TargetOnlyExperimentResult(
        output_token_ids=output,
        proposed_tokens=proposed_tokens,
        accepted_tokens=accepted_tokens,
        target_forwards=target_forwards,
        elapsed_seconds=measured_elapsed,
        acceptance_lengths=acceptance_lengths,
    )


def run_ngram_target_only_experiment(
    *,
    token_trace: list[int],
    prefix_len: int,
    max_tokens: int,
    ngram_size: int = 4,
    config: DSparkSpeculativeConfig | None = None,
    elapsed_seconds: float | None = None,
) -> TargetOnlyExperimentResult:
    if prefix_len <= 0:
        raise ValueError("prefix_len must be positive")
    if prefix_len >= len(token_trace):
        raise ValueError("prefix_len must leave at least one target token")

    generator = NGramDraftGenerator(ngram_size=ngram_size)

    def draft_fn(prefix: list[int], num_tokens: int) -> DraftProposal:
        return generator.propose(prefix, num_tokens)

    def target_fn(prefix: list[int], num_tokens: int) -> SequenceLike[int]:
        return token_trace[len(prefix):len(prefix) + num_tokens]

    return run_target_only_speculative_experiment(
        prefix_token_ids=token_trace[:prefix_len],
        max_tokens=max_tokens,
        draft_fn=draft_fn,
        target_fn=target_fn,
        config=config,
        elapsed_seconds=elapsed_seconds,
    )
