# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import bisect
import enum
import time
from collections.abc import Callable, Mapping
from functools import partial
from typing import TYPE_CHECKING, Any, Optional

import torch

from vllm.multimodal.inputs import MultiModalFeatureSpec
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams
from vllm.utils import length_from_prompt_token_ids_or_embeds
from vllm.v1.engine import (
    EngineCoreEvent,
    EngineCoreEventType,
    EngineCoreRequest,
    FinishReason,
)
from vllm.v1.structured_output.request import StructuredOutputRequest
from vllm.v1.utils import ConstantList

if TYPE_CHECKING:
    from vllm.lora.request import LoRARequest
    from vllm.v1.core.block_masking import BlockMaskingState
    from vllm.v1.core.kv_cache_utils import BlockHash


class Request:
    def __init__(
        self,
        request_id: str,
        prompt_token_ids: list[int] | None,
        sampling_params: SamplingParams | None,
        pooling_params: PoolingParams | None,
        eos_token_id: int | None,
        client_index: int = 0,
        arrival_time: float | None = None,
        prompt_embeds: torch.Tensor | None = None,
        mm_features: list[MultiModalFeatureSpec] | None = None,
        lora_request: Optional["LoRARequest"] = None,
        cache_salt: str | None = None,
        priority: int = 0,
        trace_headers: Mapping[str, str] | None = None,
        block_hasher: Callable[["Request"], list["BlockHash"]] | None = None,
    ) -> None:
        self.request_id = request_id
        self.client_index = client_index
        self.priority = priority
        self.sampling_params = sampling_params
        self.pooling_params = pooling_params
        # Because of LoRA, the eos token id can be different for each request.
        self.eos_token_id = eos_token_id
        self.lora_request = lora_request
        self.structured_output_request = StructuredOutputRequest.from_sampling_params(
            sampling_params
        )
        self.arrival_time = arrival_time if arrival_time is not None else time.time()

        self.status = RequestStatus.WAITING
        self.events: list[EngineCoreEvent] = []
        self.stop_reason: int | str | None = None

        # P/D: Connector-specific KV transfer parameters.
        self.kv_transfer_params: dict[str, Any] | None = None

        if pooling_params is not None:
            # Pooling models.
            self.max_tokens = 1
        elif sampling_params is not None:
            # Generative models.
            assert sampling_params.max_tokens is not None
            self.max_tokens = sampling_params.max_tokens
            if self.structured_output_request is not None:
                self.status = RequestStatus.WAITING_FOR_FSM

            if sampling_params.extra_args is not None:
                self.kv_transfer_params = sampling_params.extra_args.get(
                    "kv_transfer_params"
                )
        else:
            raise ValueError("sampling_params and pooling_params can't both be unset")

        self.prompt_token_ids = prompt_token_ids
        self.prompt_embeds = prompt_embeds
        self.num_prompt_tokens = length_from_prompt_token_ids_or_embeds(
            prompt_token_ids, prompt_embeds
        )
        self._output_token_ids: list[int] = []
        self._all_token_ids: list[int] = (
            self.prompt_token_ids.copy()
            if self.prompt_token_ids is not None
            else [0] * self.num_prompt_tokens
        )

        # Used in async scheduling.
        self.num_output_placeholders = 0
        # Used in forced preemption (reset_prefix_cache) with async scheduling.
        self.discard_latest_async_tokens = False

        self.spec_token_ids: list[int] = []
        self.num_computed_tokens = 0
        self.cache_salt: str | None = cache_salt

        # Multi-modal related
        self.mm_features = mm_features or []
        self.num_encoder_inputs = len(self.mm_features)
        self.has_encoder_inputs = self.num_encoder_inputs > 0

        # Read-only views
        # Prevent directly appending to these lists since
        # they should also be updated simultaneously.
        self.output_token_ids = ConstantList(self._output_token_ids)
        self.all_token_ids = ConstantList(self._all_token_ids)
        # trace_headers
        self.trace_headers = trace_headers
        # State
        # The number of tokens with prefix cache hits.
        self.num_cached_tokens = -1

        # The number of NaNs in logits. A value greater than 0
        # indicates that the output is corrupted
        self.num_nans_in_logits = 0

        # The number of requests being preempted by the scheduler
        self.num_preemptions = 0

        # The number of tokens that have been computed remotely.
        self.num_external_computed_tokens = 0

        self.block_hashes: list[BlockHash] = []
        self.get_hash_new_full_blocks: Callable[[], list[BlockHash]] | None = None
        if block_hasher is not None:
            self.get_hash_new_full_blocks = partial(block_hasher, self)
            self.block_hashes = self.get_hash_new_full_blocks()

        self.skip_reading_prefix_cache = self.get_skip_reading_prefix_cache()

        # Track spans that have been compacted (removed from KV cache).
        # These are LOGICAL positions that no longer exist in the physical cache.
        # Used to translate between logical and physical positions.
        # Updated by mask_token_span() which combines masking and compaction
        # in a single step.  Setting this property precomputes auxiliary
        # arrays for O(log n) binary-search position lookups.
        self.compacted_spans = []

        # Block masking state for Memento-style generation.
        # Initialized by the scheduler if block masking is enabled.
        self.block_masking_state: Optional["BlockMaskingState"] = None

    @classmethod
    def from_engine_core_request(
        cls,
        request: EngineCoreRequest,
        block_hasher: Callable[["Request"], list["BlockHash"]] | None,
    ) -> "Request":
        return cls(
            request_id=request.request_id,
            client_index=request.client_index,
            prompt_token_ids=request.prompt_token_ids,
            prompt_embeds=request.prompt_embeds,
            mm_features=request.mm_features,
            sampling_params=request.sampling_params,
            pooling_params=request.pooling_params,
            eos_token_id=request.eos_token_id,
            arrival_time=request.arrival_time,
            lora_request=request.lora_request,
            cache_salt=request.cache_salt,
            priority=request.priority,
            trace_headers=request.trace_headers,
            block_hasher=block_hasher,
        )

    def append_output_token_ids(
        self,
        token_ids: int | list[int],
    ) -> None:
        if isinstance(token_ids, int):
            self._output_token_ids.append(token_ids)
            self._all_token_ids.append(token_ids)
        else:
            self._output_token_ids.extend(token_ids)
            self._all_token_ids.extend(token_ids)

        if self.get_hash_new_full_blocks is not None:
            self.block_hashes.extend(self.get_hash_new_full_blocks())

    @property
    def use_structured_output(self) -> bool:
        return self.structured_output_request is not None

    @property
    def num_tokens(self) -> int:
        return len(self._all_token_ids)

    @property
    def num_tokens_with_spec(self) -> int:
        return len(self._all_token_ids) + len(self.spec_token_ids)

    @property
    def num_output_tokens(self) -> int:
        return len(self._output_token_ids)

    def get_skip_reading_prefix_cache(self) -> bool:
        if (
            self.sampling_params is not None
            and self.sampling_params.skip_reading_prefix_cache is not None
        ):
            return self.sampling_params.skip_reading_prefix_cache
        elif (
            self.pooling_params is not None
            and self.pooling_params.skip_reading_prefix_cache is not None
        ):
            return self.pooling_params.skip_reading_prefix_cache
        return False

    def is_finished(self) -> bool:
        return RequestStatus.is_finished(self.status)

    def get_finished_reason(self) -> FinishReason | None:
        return RequestStatus.get_finished_reason(self.status)

    def get_num_encoder_embeds(self, input_id: int) -> int:
        assert input_id < len(self.mm_features)
        num_embeds = self.mm_features[input_id].mm_position.get_num_embeds
        return num_embeds

    def record_event(
        self,
        event_type: EngineCoreEventType,
        timestamp: float | None = None,
    ) -> None:
        self.events.append(EngineCoreEvent.new_event(event_type, timestamp))

    def take_events(self) -> list[EngineCoreEvent] | None:
        if not self.events:
            return None
        events, self.events = self.events, []
        return events

    def __lt__(self, other: "Request") -> bool:
        """
        Compare two requests based on priority, arrival time, and request ID.
        Used in priority scheduling.
        """
        if self.priority != other.priority:
            return self.priority < other.priority
        if self.arrival_time != other.arrival_time:
            return self.arrival_time < other.arrival_time
        if self.request_id != other.request_id:
            return self.request_id < other.request_id
        return id(self) < id(other)

    @property
    def compacted_spans(self) -> list[tuple[int, int]]:
        return self._compacted_spans

    @compacted_spans.setter
    def compacted_spans(self, spans: list[tuple[int, int]]) -> None:
        self._compacted_spans = spans
        # Precompute auxiliary arrays for O(log n) position lookups.
        # _cs_starts[i] = start of span i  (for bisect in logical_to_physical)
        # _cs_cumulative[i] = total removed tokens through span i (inclusive)
        # _cs_gap_physical[i] = physical position where gap i starts
        # _cs_gap_logical[i]  = logical position where gap i starts
        starts: list[int] = []
        cumulative: list[int] = []
        gap_physical: list[int] = [0]
        gap_logical: list[int] = [0]
        total = 0
        for s, e in spans:
            starts.append(s)
            total += e - s
            cumulative.append(total)
            gap_physical.append(e - total)
            gap_logical.append(e)
        self._cs_starts = starts
        self._cs_cumulative = cumulative
        self._cs_gap_physical = gap_physical
        self._cs_gap_logical = gap_logical

    def is_token_active(self, position: int) -> bool:
        """Check if a token is active (not compacted/removed). O(log n)."""
        if not self._cs_starts:
            return True
        idx = bisect.bisect_right(self._cs_starts, position) - 1
        if idx < 0:
            return True
        return position >= self._compacted_spans[idx][1]

    def get_active_positions(self, num_tokens: int) -> list[int]:
        """Get list of active (non-compacted) logical positions.

        Iterates through gaps between sorted, non-overlapping compacted_spans
        in O(num_active + num_spans) rather than O(num_tokens × num_spans).

        Args:
            num_tokens: Total number of tokens to consider.

        Returns:
            Sorted list of logical positions that are active.
        """
        if not self.compacted_spans:
            return list(range(num_tokens))
        active: list[int] = []
        prev_end = 0
        for start, end in self.compacted_spans:
            if start >= num_tokens:
                break
            active.extend(range(prev_end, min(start, num_tokens)))
            prev_end = end
        if prev_end < num_tokens:
            active.extend(range(prev_end, num_tokens))
        return active

    def logical_to_physical(self, logical_pos: int) -> int:
        """Convert a logical position to physical position after compaction.

        Uses binary search on precomputed cumulative removed counts.
        O(log n) where n is the number of compacted spans.

        Args:
            logical_pos: The logical (original) token position.

        Returns:
            The physical position in the compacted KV cache.
        """
        if not self._cs_starts:
            return logical_pos
        # Find the rightmost span with start <= logical_pos.
        idx = bisect.bisect_right(self._cs_starts, logical_pos) - 1
        if idx < 0:
            return logical_pos
        _, end = self._compacted_spans[idx]
        if logical_pos >= end:
            return logical_pos - self._cs_cumulative[idx]
        # logical_pos is inside span (compacted position — shouldn't normally
        # be queried, but handle gracefully).
        prev_cum = self._cs_cumulative[idx - 1] if idx > 0 else 0
        return logical_pos - prev_cum - (logical_pos - self._cs_starts[idx])

    def physical_to_logical(self, physical_pos: int) -> int:
        """Convert a physical position back to logical position.

        Uses binary search on precomputed gap boundaries.
        O(log n) where n is the number of compacted spans.

        Args:
            physical_pos: The physical position in the compacted KV cache.

        Returns:
            The logical (original) token position.
        """
        if not self._cs_gap_physical:
            return physical_pos
        # Find which gap the physical position falls in.
        idx = bisect.bisect_right(self._cs_gap_physical, physical_pos) - 1
        return (self._cs_gap_logical[idx]
                + (physical_pos - self._cs_gap_physical[idx]))

    def get_compacted_token_count(self) -> int:
        """Get total number of tokens that have been compacted (physically removed)."""
        return self._cs_cumulative[-1] if self._cs_cumulative else 0


class RequestStatus(enum.IntEnum):
    """Status of a request."""

    WAITING = enum.auto()
    WAITING_FOR_FSM = enum.auto()
    WAITING_FOR_REMOTE_KVS = enum.auto()
    RUNNING = enum.auto()
    PREEMPTED = enum.auto()
    # Note: anything after PREEMPTED will be considered
    # as a finished status.
    FINISHED_STOPPED = enum.auto()
    FINISHED_LENGTH_CAPPED = enum.auto()
    FINISHED_ABORTED = enum.auto()
    FINISHED_IGNORED = enum.auto()
    FINISHED_ERROR = enum.auto()

    def __str__(self):
        return self.name

    @staticmethod
    def is_finished(status: "RequestStatus") -> bool:
        return status > RequestStatus.PREEMPTED

    @staticmethod
    def get_finished_reason(status: "RequestStatus") -> FinishReason | None:
        return _FINISHED_REASON_MAP.get(status)


# Mapping of finished statuses to their finish reasons.
# NOTE: The ignored requests are the requests whose prompt lengths
# are longer than the model's length cap. Therefore, the stop
# reason should also be "length" as in OpenAI API.
_FINISHED_REASON_MAP = {
    RequestStatus.FINISHED_STOPPED: FinishReason.STOP,
    RequestStatus.FINISHED_LENGTH_CAPPED: FinishReason.LENGTH,
    RequestStatus.FINISHED_ABORTED: FinishReason.ABORT,
    RequestStatus.FINISHED_IGNORED: FinishReason.LENGTH,
    RequestStatus.FINISHED_ERROR: FinishReason.ERROR,
}
