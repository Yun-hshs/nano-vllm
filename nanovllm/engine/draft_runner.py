import math
from types import MethodType
from time import perf_counter

import torch
from transformers import AutoConfig

from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.speculative import DraftProposal
from nanovllm.utils.context import reset_context, set_context
from nanovllm.utils.loader import load_model


class NanoDraftGenerator:
    """Native nano-vllm draft generator with its own KV cache."""

    def __init__(
        self,
        model: str,
        *,
        max_model_len: int,
        block_size: int,
        device: str = "cuda",
    ):
        self.model_path = model
        self.max_model_len = max_model_len
        self.block_size = block_size
        self.device = device
        self.cached_token_ids: list[int] = []
        self.next_logits = None
        self.last_draft_forwards = 0
        self.last_draft_rebuilds = 0
        self.last_draft_cached_steps = 0
        self.last_draft_graph_enabled = False
        self.last_draft_graph_replays = 0
        self.last_draft_prefill_ms = 0.0
        self.last_draft_cache_extend_ms = 0.0
        self.last_draft_decode_graph_ms = 0.0
        self.last_draft_decode_eager_ms = 0.0
        self.last_draft_token_select_ms = 0.0

        self.hf_config = AutoConfig.from_pretrained(model)
        self.hf_config.max_position_embeddings = min(max_model_len, self.hf_config.max_position_embeddings)
        default_dtype = torch.get_default_dtype()
        default_device = torch.get_default_device()
        torch.set_default_dtype(self.hf_config.dtype)
        torch.set_default_device(device)
        self.model = Qwen3ForCausalLM(self.hf_config)
        load_model(self.model, model)
        self._disable_torch_compile(self.model)
        self._allocate_kv_cache()
        self.decode_graph = None
        self.decode_graph_vars = None
        if str(device).startswith("cuda") and torch.cuda.is_available():
            self._capture_decode_graph()
        torch.set_default_device(default_device)
        torch.set_default_dtype(default_dtype)

    def _disable_torch_compile(self, module: torch.nn.Module):
        for child in module.modules():
            for name in ("forward", "rms_forward", "add_rms_forward"):
                method = getattr(child, name, None)
                wrapped = getattr(method, "__wrapped__", None)
                if wrapped is not None:
                    setattr(child, name, MethodType(wrapped, child))

    def _allocate_kv_cache(self):
        num_blocks = math.ceil(self.max_model_len / self.block_size) + 1
        num_kv_heads = self.hf_config.num_key_value_heads
        head_dim = getattr(self.hf_config, "head_dim", self.hf_config.hidden_size // self.hf_config.num_attention_heads)
        self.kv_cache = torch.empty(
            2,
            self.hf_config.num_hidden_layers,
            num_blocks,
            self.block_size,
            num_kv_heads,
            head_dim,
            device=self.device,
            dtype=self.hf_config.dtype,
        )
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1
        self.block_table = list(range(num_blocks))

    @torch.inference_mode()
    def _capture_decode_graph(self):
        input_ids = torch.zeros(1, dtype=torch.int64, device=self.device)
        positions = torch.zeros(1, dtype=torch.int64, device=self.device)
        slot_mapping = torch.zeros(1, dtype=torch.int32, device=self.device)
        context_lens = torch.ones(1, dtype=torch.int32, device=self.device)
        block_tables = torch.tensor([self.block_table], dtype=torch.int32, device=self.device)
        outputs = torch.empty(1, self.hf_config.hidden_size, dtype=self.hf_config.dtype, device=self.device)
        graph = torch.cuda.CUDAGraph()
        graph_pool = None

        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        for _ in range(3):
            outputs.copy_(self.model(input_ids, positions))
        torch.cuda.synchronize()
        try:
            with torch.cuda.graph(graph, graph_pool):
                outputs.copy_(self.model(input_ids, positions))
            torch.cuda.synchronize()
        except Exception:
            reset_context()
            self.decode_graph = None
            self.decode_graph_vars = None
            return
        reset_context()
        self.decode_graph = graph
        self.decode_graph_vars = {
            "input_ids": input_ids,
            "positions": positions,
            "slot_mapping": slot_mapping,
            "context_lens": context_lens,
            "block_tables": block_tables,
            "outputs": outputs,
        }

    def _reset_last_stats(self):
        self.last_draft_forwards = 0
        self.last_draft_rebuilds = 0
        self.last_draft_cached_steps = 0
        self.last_draft_graph_enabled = self.decode_graph is not None
        self.last_draft_graph_replays = 0
        self.last_draft_prefill_ms = 0.0
        self.last_draft_cache_extend_ms = 0.0
        self.last_draft_decode_graph_ms = 0.0
        self.last_draft_decode_eager_ms = 0.0
        self.last_draft_token_select_ms = 0.0

    @torch.inference_mode()
    def propose(self, prefix_token_ids: list[int], num_tokens: int) -> DraftProposal:
        if num_tokens <= 0:
            return DraftProposal([])
        self._reset_last_stats()
        logits = self._ensure_cache(prefix_token_ids)
        proposed_token_ids: list[int] = []
        for _ in range(num_tokens):
            token_id = self._last_token_id(logits)
            proposed_token_ids.append(token_id)
            logits = self._decode_one(token_id)
        return DraftProposal(proposed_token_ids, [1.0] * len(proposed_token_ids))

    def _last_logits(self, logits: torch.Tensor) -> torch.Tensor:
        if logits.dim() == 3:
            return logits[:, -1, :]
        return logits

    def _last_token_id(self, logits: torch.Tensor) -> int:
        token_select_start = perf_counter()
        token_id = int(torch.argmax(self._last_logits(logits), dim=-1).item())
        self.last_draft_token_select_ms += (perf_counter() - token_select_start) * 1000
        return token_id

    def _ensure_cache(self, prefix_token_ids: list[int]):
        prefix = list(prefix_token_ids)
        if self.next_logits is not None and self.cached_token_ids == prefix:
            self.last_draft_cached_steps += len(prefix)
            return self.next_logits
        if (
            self.next_logits is not None
            and self.cached_token_ids
            and prefix[:len(self.cached_token_ids)] == self.cached_token_ids
        ):
            cached_len = len(self.cached_token_ids)
            self.last_draft_cached_steps += cached_len
            cache_extend_start = perf_counter()
            for token_id in prefix[cached_len:]:
                self._decode_one(token_id)
            self.last_draft_cache_extend_ms += (perf_counter() - cache_extend_start) * 1000
            return self.next_logits
        return self._prefill(prefix)

    def _prefill(self, token_ids: list[int]):
        prefill_start = perf_counter()
        input_ids = torch.tensor(token_ids, dtype=torch.int64, device=self.device)
        positions = torch.arange(len(token_ids), dtype=torch.int64, device=self.device)
        cu_seqlens_q = torch.tensor([0, len(token_ids)], dtype=torch.int32, device=self.device)
        cu_seqlens_k = torch.tensor([0, len(token_ids)], dtype=torch.int32, device=self.device)
        slot_mapping = torch.arange(len(token_ids), dtype=torch.int32, device=self.device)
        set_context(True, cu_seqlens_q, cu_seqlens_k, len(token_ids), len(token_ids), slot_mapping, None, None)
        self.next_logits = self.model.compute_logits(self.model(input_ids, positions))
        reset_context()
        self.cached_token_ids = list(token_ids)
        self.last_draft_forwards += 1
        self.last_draft_rebuilds += 1
        self.last_draft_prefill_ms += (perf_counter() - prefill_start) * 1000
        return self.next_logits

    def _decode_one(self, token_id: int):
        position_id = len(self.cached_token_ids)
        if self.decode_graph is not None:
            return self._decode_one_graph(token_id, position_id)
        return self._decode_one_eager(token_id, position_id)

    def _decode_one_graph(self, token_id: int, position_id: int):
        decode_start = perf_counter()
        graph_vars = self.decode_graph_vars
        graph_vars["input_ids"][0] = token_id
        graph_vars["positions"][0] = position_id
        graph_vars["slot_mapping"][0] = position_id
        graph_vars["context_lens"][0] = position_id + 1
        set_context(
            False,
            slot_mapping=graph_vars["slot_mapping"],
            context_lens=graph_vars["context_lens"],
            block_tables=graph_vars["block_tables"],
        )
        graph = self.decode_graph
        graph.replay()
        reset_context()
        self.next_logits = self.model.compute_logits(graph_vars["outputs"])
        self.cached_token_ids.append(token_id)
        self.last_draft_forwards += 1
        self.last_draft_graph_replays += 1
        self.last_draft_decode_graph_ms += (perf_counter() - decode_start) * 1000
        return self.next_logits

    def _decode_one_eager(self, token_id: int, position_id: int):
        decode_start = perf_counter()
        input_ids = torch.tensor([token_id], dtype=torch.int64, device=self.device)
        positions = torch.tensor([position_id], dtype=torch.int64, device=self.device)
        slot_mapping = torch.tensor([position_id], dtype=torch.int32, device=self.device)
        context_lens = torch.tensor([position_id + 1], dtype=torch.int32, device=self.device)
        block_tables = torch.tensor([self.block_table], dtype=torch.int32, device=self.device)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        self.next_logits = self.model.compute_logits(self.model(input_ids, positions))
        reset_context()
        self.cached_token_ids.append(token_id)
        self.last_draft_forwards += 1
        self.last_draft_decode_eager_ms += (perf_counter() - decode_start) * 1000
        return self.next_logits
