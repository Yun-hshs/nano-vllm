import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.draft_runner import NanoDraftGenerator
from nanovllm.speculative import HFDraftGenerator, NGramDraftGenerator, SpeculativeStats, should_use_greedy_verification


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self.speculative_config = config.speculative_config
        self.speculative_stats = SpeculativeStats()
        if config.speculative_config.enabled and config.speculative_config.draft_type == "hf":
            self.draft_generator = HFDraftGenerator(
                config.speculative_config.draft_model,
                device=config.speculative_config.draft_device,
                dtype=config.speculative_config.draft_dtype,
            )
        elif config.speculative_config.enabled and config.speculative_config.draft_type == "nano":
            self.draft_generator = None
        else:
            self.draft_generator = NGramDraftGenerator(config.speculative_config.ngram_size)
        Sequence.block_size = config.kvcache_block_size
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        if config.speculative_config.enabled and config.speculative_config.draft_type == "nano":
            self.draft_generator = NanoDraftGenerator(
                config.speculative_config.draft_model,
                max_model_len=config.max_model_len,
                block_size=config.kvcache_block_size,
                device=config.speculative_config.draft_device,
            )
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self):
        if self.speculative_config.enabled:
            return self._step_speculative()
        return self._step_regular()

    def _step_regular(self):
        seqs, is_prefill = self.scheduler.schedule()
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids, is_prefill)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        return outputs, num_tokens

    def _step_speculative(self):
        if (
            self.scheduler.waiting
            or len(self.scheduler.running) != 1
            or self.speculative_config.draft_type not in ("ngram", "hf", "nano")
            or self.model_runner.world_size != 1
        ):
            return self._step_regular()

        seq = self.scheduler.running[0]
        if seq.is_prefill:
            return self._step_regular()

        remaining_tokens = seq.max_tokens - seq.num_completion_tokens
        step_start = perf_counter()
        draft_propose_start = perf_counter()
        proposal = self.draft_generator.propose(
            seq.token_ids,
            min(self.speculative_config.num_speculative_tokens, remaining_tokens),
        )
        draft_propose_ms = (perf_counter() - draft_propose_start) * 1000
        draft_forwards = getattr(self.draft_generator, "last_draft_forwards", 0)
        draft_rebuilds = getattr(self.draft_generator, "last_draft_rebuilds", 0)
        draft_cached_steps = getattr(self.draft_generator, "last_draft_cached_steps", 0)
        draft_graph_enabled = getattr(self.draft_generator, "last_draft_graph_enabled", False)
        draft_graph_replays = getattr(self.draft_generator, "last_draft_graph_replays", 0)
        draft_prefill_ms = getattr(self.draft_generator, "last_draft_prefill_ms", 0.0)
        draft_cache_extend_ms = getattr(self.draft_generator, "last_draft_cache_extend_ms", 0.0)
        draft_decode_graph_ms = getattr(self.draft_generator, "last_draft_decode_graph_ms", 0.0)
        draft_decode_eager_ms = getattr(self.draft_generator, "last_draft_decode_eager_ms", 0.0)
        draft_token_select_ms = getattr(self.draft_generator, "last_draft_token_select_ms", 0.0)
        if not proposal.token_ids:
            outputs, num_tokens = self._step_regular()
            emitted_tokens = -num_tokens if num_tokens < 0 else 0
            self.speculative_stats.record_step(
                proposed_tokens=0,
                accepted_tokens=0,
                draft_forwards=draft_forwards,
                draft_rebuilds=draft_rebuilds,
                draft_cached_steps=draft_cached_steps,
                draft_graph_enabled=draft_graph_enabled,
                draft_graph_replays=draft_graph_replays,
                target_forwards=1 if num_tokens < 0 else 0,
                emitted_tokens=emitted_tokens,
                draft_propose_ms=draft_propose_ms,
                draft_prefill_ms=draft_prefill_ms,
                draft_cache_extend_ms=draft_cache_extend_ms,
                draft_decode_graph_ms=draft_decode_graph_ms,
                draft_decode_eager_ms=draft_decode_eager_ms,
                draft_token_select_ms=draft_token_select_ms,
                speculative_step_ms=(perf_counter() - step_start) * 1000,
            )
            return outputs, num_tokens

        if self._crosses_speculative_block_boundary(seq, len(proposal.token_ids)):
            outputs, num_tokens = self._step_regular()
            emitted_tokens = -num_tokens if num_tokens < 0 else 0
            self.speculative_stats.record_step(
                proposed_tokens=len(proposal.token_ids),
                accepted_tokens=0,
                draft_forwards=draft_forwards,
                draft_rebuilds=draft_rebuilds,
                draft_cached_steps=draft_cached_steps,
                draft_graph_enabled=draft_graph_enabled,
                draft_graph_replays=draft_graph_replays,
                target_forwards=1 if num_tokens < 0 else 0,
                emitted_tokens=emitted_tokens,
                draft_propose_ms=draft_propose_ms,
                draft_prefill_ms=draft_prefill_ms,
                draft_cache_extend_ms=draft_cache_extend_ms,
                draft_decode_graph_ms=draft_decode_graph_ms,
                draft_decode_eager_ms=draft_decode_eager_ms,
                draft_token_select_ms=draft_token_select_ms,
                speculative_step_ms=(perf_counter() - step_start) * 1000,
            )
            return outputs, num_tokens

        old_num_tokens = len(seq)
        target_verify_start = perf_counter()
        target_token_ids = self.model_runner.call("run_speculative", seq, proposal.token_ids)
        target_verify_ms = (perf_counter() - target_verify_start) * 1000
        target_forward_ms = getattr(self.model_runner, "last_speculative_forward_ms", 0.0)
        target_argmax_ms = getattr(self.model_runner, "last_speculative_argmax_ms", 0.0)
        target_compare_start = perf_counter()
        accepted_tokens = 0
        for draft_token_id, target_token_id in zip(proposal.token_ids, target_token_ids):
            if draft_token_id != target_token_id:
                break
            accepted_tokens += 1
            if accepted_tokens == remaining_tokens:
                break
        target_compare_ms = (perf_counter() - target_compare_start) * 1000

        emitted_token_ids = proposal.token_ids[:accepted_tokens]
        if len(emitted_token_ids) < remaining_tokens and accepted_tokens < len(target_token_ids):
            emitted_token_ids.append(target_token_ids[accepted_tokens])

        append_start = perf_counter()
        emitted_tokens = self._append_speculative_tokens(
            seq,
            emitted_token_ids,
            cached_tokens=old_num_tokens + accepted_tokens,
        )
        append_tokens_ms = (perf_counter() - append_start) * 1000
        target_greedy_forwards = int(should_use_greedy_verification(
            seq.temperature,
            self.speculative_config.greedy_verification_temperature,
        ))
        outputs = [(seq.seq_id, seq.completion_token_ids)] if seq.is_finished else []
        self.speculative_stats.record_step(
            proposed_tokens=len(proposal.token_ids),
            accepted_tokens=accepted_tokens,
            draft_forwards=draft_forwards,
            draft_rebuilds=draft_rebuilds,
            draft_cached_steps=draft_cached_steps,
            draft_graph_enabled=draft_graph_enabled,
            draft_graph_replays=draft_graph_replays,
            target_forwards=1,
            target_greedy_forwards=target_greedy_forwards,
            target_greedy_tokens=len(target_token_ids) if target_greedy_forwards else 0,
            emitted_tokens=emitted_tokens,
            draft_propose_ms=draft_propose_ms,
            draft_prefill_ms=draft_prefill_ms,
            draft_cache_extend_ms=draft_cache_extend_ms,
            draft_decode_graph_ms=draft_decode_graph_ms,
            draft_decode_eager_ms=draft_decode_eager_ms,
            draft_token_select_ms=draft_token_select_ms,
            target_verify_ms=target_verify_ms,
            target_forward_ms=target_forward_ms,
            target_argmax_ms=target_argmax_ms,
            target_compare_ms=target_compare_ms,
            append_tokens_ms=append_tokens_ms,
            speculative_step_ms=(perf_counter() - step_start) * 1000,
        )
        return outputs, -emitted_tokens

    def _crosses_speculative_block_boundary(self, seq: Sequence, num_draft_tokens: int) -> bool:
        return len(seq) + num_draft_tokens > len(seq.block_table) * seq.block_size

    def _append_speculative_tokens(
        self,
        seq: Sequence,
        token_ids: list[int],
        *,
        cached_tokens: int,
    ) -> int:
        emitted_tokens = 0
        for token_id in token_ids:
            seq.append_token(token_id)
            emitted_tokens += 1
            if (not seq.ignore_eos and token_id == self.scheduler.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.scheduler.block_manager.deallocate(seq)
                if seq in self.scheduler.running:
                    self.scheduler.running.remove(seq)
                return emitted_tokens
        seq.num_cached_tokens = min(cached_tokens, seq.num_tokens)
        seq.num_scheduled_tokens = 0
        return emitted_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if num_tokens > 0:
                prefill_throughput = num_tokens / (perf_counter() - t)
            else:
                decode_throughput = -num_tokens / (perf_counter() - t)
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                pbar.update(1)
        pbar.close()
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs
