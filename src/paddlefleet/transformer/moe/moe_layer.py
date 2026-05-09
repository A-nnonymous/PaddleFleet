# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import hashlib
import logging
import os
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING

import paddle
from paddle import framework, nn
from paddle.autograd import PyLayer
from paddle.distributed.fleet.utils.sequence_parallel_utils import (
    GatherOp,
    ScatterOp,
)

import paddlefleet

if TYPE_CHECKING:
    from paddle.distributed.fleet.meta_parallel import LayerSpec

    from paddlefleet.process_groups_config import ProcessGroupCollection
    from paddlefleet.transformer.transformer_config import TransformerConfig

from paddlefleet import utils
from paddlefleet.transformer.utils import profile

from .fp8_utils import fused_stack_quant_without_cache
from .fused_a2a import configure_buffer
from .fusion_layer_utils import FusionMoePyLayer
from .moe_expert import GroupedMLPExpert, StandardMLPExpert
from .moe_router import TopKRouter
from .moe_shared_expert import StandardMLPSharedExpert
from .moe_utils import AddAuxiliaryLoss
from .token_dispatcher import AllToAllTokenDispatcher, MoEFlexTokenDispatcher

logger = logging.getLogger(__name__)

# Alignment switch and MD5 logging for MoE precision debugging
_ERNIECORE_ALIGNMENT = (
    os.environ.get("gpt_model_use_experimental_version", "0") == "1"
)
_LOG_LAYER_MD5 = os.environ.get("LOG_LAYER_MD5", "0") == "1"


# Intermediate-result dump for precision diff debugging
# When MOE_DEBUG_DUMP=1, MoELayer stores snapshots of forward tensors
# and registers backward hooks so the test (or any caller) can compare
# corresponding intermediates across variants (baseline / BF16 / FP8 ...).
# Read at instance-construction time so tests can toggle via env var.
def _moe_debug_dump_enabled() -> bool:
    return os.environ.get("MOE_DEBUG_DUMP", "0") == "1"


def _log_moe_md5(tensor, name, layer_idx=None):
    """Log MD5 of a tensor for MoE precision alignment debugging."""
    if _LOG_LAYER_MD5 and _ERNIECORE_ALIGNMENT:
        from paddlefleet.transformer.transformer_layer import TransformerLayer

        if TransformerLayer._skip_mtp_probes:
            return  # Skip MTP passes — EC has no MTP
        data = tensor.detach().cast("float32").numpy().tobytes()
        md5 = hashlib.md5(data).hexdigest()
        rank = (
            paddle.distributed.get_rank()
            if paddle.distributed.is_initialized()
            else 0
        )
        layer_str = f" Layer={layer_idx}" if layer_idx is not None else ""
        print(
            f"[MD5 MoE] Rank={rank}{layer_str} {name} MD5={md5} shape={list(tensor.shape)}",
            flush=True,
        )


if paddlefleet.ops.is_sonic_moe_available():
    from paddlefleet.ops.sonicmoe.enums import ActivationType
    from paddlefleet.ops.sonicmoe.ernie_compat.deepep_metadata import (
        deepep_topk_to_sonic_metadata,
    )
    from paddlefleet.ops.sonicmoe.ernie_compat.mlp_node_v2 import (
        _differentiable_router_scores,
    )
    from paddlefleet.ops.sonicmoe.functional import (
        _DownProjection,
        _refresh_fp8_config,
        _UpProjection,
    )
    from paddlefleet.ops.sonicmoe.functional.utils import enable_fp8

from .moe_utils import (
    global_moe_balance_training_logs_enabled,
    log_moe_balance,
    permute,
    unpermute,
)


class GradDtypeGuard(PyLayer):
    """Guard the grad's dtype if different from input's dtype."""

    @staticmethod
    def forward(ctx, x, dtype):
        """forward"""
        return paddle.empty([0], dtype=dtype), {"x": x}

    @staticmethod
    def backward(ctx, grad):
        """backward"""
        return grad


class GradDtypeUnguard(PyLayer):
    """Remove grad dtype guard."""

    @staticmethod
    def forward(ctx, x, status):
        """forward"""
        if hasattr(ctx, "set_grad_in_dtype_consistent"):
            ctx.set_grad_in_dtype_consistent(False)
        return status["x"]

    @staticmethod
    def backward(ctx, grad):
        """backward"""
        return grad


@dataclass
class MoESublayers:
    """MoE Layer Sublayers spec"""

    mlp_spec: LayerSpec | type = None  # Used by experts


class MoELayer(nn.Layer):
    def __init__(
        self,
        config: TransformerConfig,
        sublayers: MoESublayers | None = None,
        pg_collection: ProcessGroupCollection | None = None,
    ):
        super().__init__()
        # Debug dump buffers: populated during forward and backward when
        # MOE_DEBUG_DUMP=1.  Always-allocated (cheap when disabled because
        # no tensors ever get inserted); callers should snapshot via
        # `self._debug_dump` and then call `_dbg_reset()` between steps.
        self._debug_dump_enabled = _moe_debug_dump_enabled()
        self._debug_dump = {"fwd": {}, "bwd": {}}
        self.config = config
        self.moe_sublayers = sublayers
        routed_expert_config = deepcopy(config)
        shared_expert_config = deepcopy(config)
        self.pg_collection = pg_collection
        self.hidden_size = config.hidden_size
        self.moe_intermediate_size = config.moe_intermediate_size
        self.num_experts = config.n_routed_experts
        self.n_shared_experts = config.n_shared_experts
        self.moe_shared_expert_intermediate_size = None
        if self.n_shared_experts:
            self.moe_shared_expert_intermediate_size = (
                self.moe_intermediate_size * self.n_shared_experts
            )
        self.num_experts_per_tok = config.num_experts_per_tok
        self.hidden_act = config.hidden_act
        self.sequence_parallel = config.sequence_parallel
        self.tensor_model_parallel_size = config.tensor_model_parallel_size
        self.moe_token_dispatcher_type = config.moe_token_dispatcher_type
        self.moe_shared_expert_overlap = config.moe_shared_expert_overlap
        self.fp8 = config.fp8
        self.fp8_dispatch = bool(config.fp8)
        self.fp8_wgrad = config.fp8_wgrad
        self.dw_p2p_overlap = getattr(config, "dw_p2p_overlap", False)
        self.using_sonic_moe = self.config.using_sonic_moe
        self.fp8_dispatch = bool(config.fp8) and not self.using_sonic_moe
        self.fp8_wgrad = config.fp8_wgrad
        self.moe_expert_fusion = config.moe_expert_fusion
        self.moe_subbatch_token_num_after_dispatch = (
            config.moe_subbatch_token_num_after_dispatch
        )
        if self.using_sonic_moe:
            assert paddlefleet.ops.is_sonic_moe_available(), (
                paddlefleet.ops.blocked_import_messages[
                    "paddlefleet.ops.sonicmoe"
                ]
            )
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.moe_grouped_gemm = config.moe_grouped_gemm
        self.moe_deep_gemm = config.moe_deep_gemm

        if self.moe_deep_gemm:
            incompatible_reasons = []
            if not self.moe_grouped_gemm:
                incompatible_reasons.append("moe_grouped_gemm must be True")
            if incompatible_reasons:
                logging.warning(
                    "moe_deep_gemm=True is ignored because %s; "
                    "setting moe_deep_gemm to False.",
                    " and ".join(incompatible_reasons),
                )
                self.moe_deep_gemm = False
        self.moe_ep_barrier = config.moe_ep_barrier
        self.moe_group = pg_collection.ep
        self.expert_model_parallel_size = (
            utils.get_pg_size(self.moe_group)
            if self.moe_group is not None
            else 1
        )
        self.num_local_experts = (
            self.num_experts // self.expert_model_parallel_size
        )
        # MoE-Related Configs
        self._init_expert_parallel()

        self.gate = TopKRouter(config=config, pg_collection=pg_collection)

        self.expert_class = StandardMLPExpert
        self.shared_expert_class = StandardMLPSharedExpert

        if (
            self.expert_model_parallel_size <= 1
            and self.sequence_parallel
            and self.tensor_model_parallel_size > 1
        ):
            routed_expert_config.sequence_parallel = False
            shared_expert_config.sequence_parallel = False
        elif (
            self.expert_model_parallel_size > 1
            and self.tensor_model_parallel_size >= 1
            or paddle.version.cuda() == "12.6"
        ):
            routed_expert_config.tensor_model_parallel_size = 1

        if (
            paddle.is_compiled_with_cuda()
            and paddle.device.get_device_capability()[0] < 9
        ):
            # TODO: Support Ampere architecture after upgrade deepep in paddlepaddle
            if self.moe_token_dispatcher_type == "deepep":
                logger.info(
                    "deepep in paddlepaddle does not support compute capability < 9.0, "
                    "fallback to alltoall token dispatcher."
                )
                self.moe_token_dispatcher_type = "alltoall"
            if self.moe_deep_gemm:
                logger.warning(
                    "moe_deep_gemm is not supported when device capability < 9.0."
                )
                self.moe_deep_gemm = False

        self.moe_use_fusion_node = False
        self.moe_use_fusion_node = config.moe_use_fusion_node
        if self.expert_model_parallel_size > 1:
            if self.moe_token_dispatcher_type != "deepep":
                self.fp8_dispatch = False
                assert self.moe_grouped_gemm is False, (
                    "moe_grouped_gemm is only supported when moe_token_dispatcher_type is 'deepep' and on GPU architecture SM90 or higher. If these conditions are not met, please set it to false in the configuration yaml."
                )

        if self.fp8:
            if paddle.version.cuda() == "12.6":
                raise NotImplementedError(
                    "fp8 is not supported when cuda version == 12.6."
                )
            assert self.moe_use_fusion_node, (
                "fp8 can only be used when moe_use_fusion_node = True."
            )
            # assert not self.using_sonic_moe, (
            #     "fp8 and sonic_moe cannot be used at the same time."
            # )

        expert_args = {}
        expert_args["config"] = routed_expert_config
        expert_args["moe_intermediate_size"] = self.moe_intermediate_size
        expert_args["is_expert"] = True
        expert_args["mlp_spec"] = self.moe_sublayers.mlp_spec

        self.grouped_gemm_experts = None
        self.experts = None
        if self.moe_grouped_gemm:
            if self.fp8:
                assert self.using_sonic_moe or self.moe_deep_gemm, (
                    "For fp8 grouped_gemm, either set using_sonic_moe=True or moe_deep_gemm=True."
                )
            self.grouped_gemm_experts = GroupedMLPExpert(
                self.num_local_experts,
                routed_expert_config,
                self.moe_deep_gemm,
                pg_collection,
            )
            # Forward a dump callback into the baseline expert so that its
            # internal fc1_output / intermediate_parallel / fc2_output can
            # be snapshotted under the same key names used by sonic-moe
            # (sonic_y1 / sonic_z / sonic_down_out) for per-step diff.
            self.grouped_gemm_experts._moe_dbg = self._dbg_dump
        else:
            self.experts = nn.LayerList([])
            for i in range(self.num_experts):
                if i // self.num_experts_per_device == self.moe_rank:
                    self.experts.append(self.expert_class(**expert_args))
                else:
                    self.experts.append(None)

        shared_expert_args = deepcopy(expert_args)
        shared_expert_args["config"].hidden_size = self.config.hidden_size
        shared_expert_args["moe_intermediate_size"] = (
            self.moe_shared_expert_intermediate_size
        )
        shared_expert_args["is_expert"] = False
        if self.n_shared_experts > 0:
            self.shared_experts = self.shared_expert_class(**shared_expert_args)
        else:
            self.shared_experts = None

        if self.expert_model_parallel_size > 1:
            if self.moe_token_dispatcher_type == "deepep":
                self.token_dispatcher = MoEFlexTokenDispatcher(
                    self.num_experts_per_device,
                    self.num_experts_per_tok,
                    self.num_experts,
                    self.moe_group,
                    self.moe_ep_barrier,
                )
                if getattr(config, "deepep_buffer_configs", None) is not None:
                    configure_buffer(**config.deepep_buffer_configs)
            elif self.moe_token_dispatcher_type == "alltoall":
                local_expert_indices = list(
                    range(
                        self.moe_rank * self.num_experts_per_device,
                        (self.moe_rank + 1) * self.num_experts_per_device,
                    )
                )
                self.token_dispatcher = AllToAllTokenDispatcher(
                    self.moe_group,
                    self.expert_model_parallel_size,
                    self.num_experts_per_device,
                    local_expert_indices,
                )
            else:
                raise NotImplementedError(
                    f"Unsupported moe_token_dispatcher_type {self.moe_token_dispatcher_type}"
                )

        self.recompute_moe_gate_up = getattr(
            self.config, "recompute_moe_gate_up", False
        ) or (
            self.config.recompute_granularity == "selective"
            and self.config.recompute_modules is not None
            and "moe_gate_up" in self.config.recompute_modules
        )
        self.recompute_moe_premute = getattr(
            self.config, "recompute_moe_premute", False
        ) or (
            self.config.recompute_granularity == "selective"
            and self.config.recompute_modules is not None
            and "moe_premute" in self.config.recompute_modules
        )
        self.use_auto_subbatch = getattr(
            self.config, "use_auto_subbatch", False
        )
        self.moe_subbatch_diag = getattr(
            self.config, "moe_subbatch_diag", False
        )

        if self.expert_model_parallel_size > 1:
            self.is_mp_moe = False
            self.is_ep_moe = True
            if self.grouped_gemm_experts is not None:
                for p in self.grouped_gemm_experts.parameters():
                    p.is_moe_param = True
                    p.color = {
                        "color": "moe_expert",
                        "group": self.moe_grad_group,
                    }
                    p.no_sync = not self.is_mp_moe
                    p.expert = not self.is_mp_moe
                    if self.is_mp_moe or self.is_ep_moe:
                        p.is_distributed = True
            else:
                assert self.experts is not None, (
                    "experts should be initialized."
                )
                for p in self.experts.parameters():
                    p.is_moe_param = True
                    p.color = {
                        "color": "moe_expert",
                        "group": self.moe_grad_group,
                    }
                    p.no_sync = not self.is_mp_moe
                    p.expert = not self.is_mp_moe
                    if self.is_mp_moe or self.is_ep_moe:
                        p.is_distributed = True

    # ──────────────────────────── debug dump ────────────────────────────
    def _dbg_reset(self):
        """Clear debug dump buffers (keeps dict identities so hook closures
        registered in previous forwards are harmless after GC)."""
        if self._debug_dump_enabled:
            self._debug_dump["fwd"].clear()
            self._debug_dump["bwd"].clear()

    def _dbg_dump(self, name, tensor, grad_hook=True):
        """Snapshot `tensor` as forward intermediate `name`; optionally
        register a backward hook that saves the grad as `bwd[name]`.

        Safe to call unconditionally — becomes a no-op when disabled or
        when `tensor` is None.
        """
        if not self._debug_dump_enabled or tensor is None:
            return
        try:
            self._debug_dump["fwd"][name] = tensor.detach().clone()
        except Exception:
            # Some tensors may be non-copyable (e.g. fp8 raw buffers);
            # skip silently rather than break the forward pass.
            pass
        if grad_hook and not getattr(tensor, "stop_gradient", True):
            store = self._debug_dump["bwd"]

            def _hook(grad, n=name, s=store):
                try:
                    s[n] = grad.detach().clone()
                except Exception:
                    pass

            try:
                tensor.register_hook(_hook)
            except Exception:
                pass

    def _init_expert_parallel(self):
        def _parse_moe_expert_parallel(
            num_experts: int, expert_model_parallel_size: int
        ) -> int:
            """
            Args:
                num_experts: Total number of experts
                expert_model_parallel_size: Expert parallel groups

            Returns:
                n_routed_experts_per_device: Number of experts per device
            """
            assert num_experts >= expert_model_parallel_size, (
                f"expert num_experts={num_experts} >= moe_world_size={expert_model_parallel_size}"
            )
            assert num_experts % expert_model_parallel_size == 0, (
                f"expert num_experts={num_experts} % moe_world_size={expert_model_parallel_size} == 0"
            )

            n_routed_experts_per_device = (
                num_experts // expert_model_parallel_size
            )
            return n_routed_experts_per_device

        if self.expert_model_parallel_size > 1:
            self.moe_grad_group = self.pg_collection.expt_dp
            self.moe_rank = utils.get_pg_rank(self.moe_group)
            self.moe_rank = max(self.moe_rank, 0)
            self.num_experts_per_device = _parse_moe_expert_parallel(
                self.num_experts, self.expert_model_parallel_size
            )
        else:
            self.moe_group = None
            self.moe_rank = 0
            self.expert_model_parallel_size = 1
            self.num_experts_per_device = self.num_experts

    def expert_forward(
        self,
        dispatched_input,
        tokens_per_expert,
    ):
        outputs = []
        tokens_per_expert = (
            tokens_per_expert.tolist()
            if not isinstance(tokens_per_expert, list)
            else tokens_per_expert
        )
        chunks = paddle.split(
            dispatched_input, num_or_sections=tokens_per_expert, axis=0
        )
        for i, chunk in enumerate(chunks):
            if tokens_per_expert[i] == 0:
                continue
            chunk = chunk.contiguous()
            current_expert_idx = i + self.moe_rank * self.num_experts_per_device
            expert = self.experts[current_expert_idx]
            outputs += [expert(chunk)[0]]

        if not outputs:
            return dispatched_input

        return paddle.concat(outputs, axis=0)

    def dispatch(
        self,
        hidden_states: paddle.Tensor,
        probs: paddle.Tensor,
        routing_map: paddle.Tensor,
        async_finish: bool = False,
    ):
        hidden_states = self.token_dispatcher.dispatch_preprocess(
            hidden_states, probs, routing_map
        )
        hidden_states, fp8_dispatched_handle = (
            self.token_dispatcher.token_dispatch(
                hidden_states,
                self.fp8_dispatch,
                async_finish=async_finish,
            )
        )
        return hidden_states, fp8_dispatched_handle

    def permute(self, hidden_states: paddle.Tensor):
        global_input_tokens, tokens_per_expert = (
            self.token_dispatcher.dispatch_postprocess(hidden_states)
        )
        return global_input_tokens, tokens_per_expert

    def unpermute(self, hidden_states: paddle.Tensor):
        return self.token_dispatcher.combine_preprocess(hidden_states)

    def combine(self, hidden_states: paddle.Tensor, async_finish: bool = False):
        hidden_states = self.token_dispatcher.token_combine(
            hidden_states, async_finish=async_finish
        )
        return self.token_dispatcher.combine_postprocess(hidden_states)

    def routed_experts_compute(
        self,
        hidden_states: paddle.Tensor,
    ):
        global_input_tokens, tokens_per_expert = self.permute(hidden_states)
        expert_outs = self.expert_forward(
            global_input_tokens,
            tokens_per_expert,
        )
        return self.unpermute(expert_outs)

    # MoE forward: dispatch -> permute -> compute ->unpermute -> combine
    def custom_forward(
        self,
        hidden_states: paddle.Tensor,
        probs: paddle.Tensor,
        routing_map: paddle.Tensor,
    ):
        should_log_balance = framework._dygraph_tracer()._has_grad
        with profile("dispatch"):
            hidden_states, _ = self.dispatch(hidden_states, probs, routing_map)
        if should_log_balance and global_moe_balance_training_logs_enabled():
            log_moe_balance(
                self.layer_number,
                self.moe_group,
                self.num_experts_per_tok,
                self.token_dispatcher._comm_manager.tokens_per_expert,
            )
        with profile("fusion_mlp"):
            hidden_states = self.routed_experts_compute(hidden_states)
        with profile("combine"):
            hidden_states = self.combine(hidden_states)
        return hidden_states

    def fusion_moe_forward(
        self,
        hidden_states: paddle.Tensor,
        probs: paddle.Tensor,
        routing_map: paddle.Tensor,
        combine_overlap_handle: dict,
    ):
        # TODO(deepllz): add fp8 dispatch config && implementation
        should_log_balance = framework._dygraph_tracer()._has_grad
        with profile("dispatch"):
            dispatched_hidden_states, fp8_dispatched_handle = self.dispatch(
                hidden_states, probs, routing_map
            )
        if should_log_balance and global_moe_balance_training_logs_enabled():
            log_moe_balance(
                self.layer_number,
                self.moe_group,
                self.num_experts_per_tok,
                self.token_dispatcher._comm_manager.tokens_per_expert,
            )
        dispatched_indices = (
            self.token_dispatcher._comm_manager.dispatched_indices
        )
        dispatched_probs = self.token_dispatcher._comm_manager.dispatched_probs

        with profile("fusion_mlp"):
            if self.using_sonic_moe:
                # T = dispatched_hidden_states.shape[0]
                # K = self.num_experts_per_tok
                # stream_id = paddle.device.cuda.current_stream().cuda_stream
                # topk_scores = filter_scores(
                #     dispatched_probs,
                #     dispatched_indices,
                # )
                use_fp8 = self.fp8 is not None
                hidden_states = self.run_sonic_moe(
                    dispatched_hidden_states,
                    dispatched_indices,
                    dispatched_probs,
                    use_fp8,
                )
                # expert_frequency, expert_frequency_offset = count_cumsum(
                #     dispatched_indices,
                #     self.num_experts_per_device,
                #     do_cumsum=True,
                # )
                # activation_type = ActivationType("swiglu")

                # (
                #     expert_frequency_offset,
                #     x_gather_idx,
                #     s_scatter_idx,
                #     s_reverse_scatter_idx,
                #     num_activated_expert_per_token_offset,
                # ) = fused_expert_parallel_TC_topk_router_metadata(
                #     dispatched_indices,
                #     expert_frequency_offset,
                #     K,
                # )

                # TK = s_scatter_idx.shape[0]
                # is_varlen_K = True
                # w1 = self.grouped_gemm_experts.weight1
                # y1, z = _UpProjection.apply(
                #     dispatched_hidden_states,
                #     w1.permute(1, 2, 0),
                #     None,
                #     expert_frequency_offset,
                #     TK,
                #     K,
                #     stream_id,
                #     x_gather_idx,
                #     s_scatter_idx,
                #     s_reverse_scatter_idx,
                #     num_activated_expert_per_token_offset,
                #     is_varlen_K,
                #     activation_type,
                #     is_inference_mode_enabled=False,
                # )

                # w2 = self.grouped_gemm_experts.weight2
                # hidden_states = _DownProjection.apply(
                #     y1,
                #     z,
                #     w2.permute(1, 2, 0),
                #     None,
                #     topk_scores,
                #     expert_frequency_offset,
                #     T,
                #     K,
                #     stream_id,
                #     x_gather_idx,
                #     s_scatter_idx,
                #     s_reverse_scatter_idx,
                #     num_activated_expert_per_token_offset,
                #     is_varlen_K,
                #     activation_type,
                # )
            else:
                hidden_states = FusionMoePyLayer.apply(
                    dispatched_hidden_states,
                    dispatched_probs,
                    dispatched_indices,
                    self,
                    self.num_experts_per_tok,
                    use_fp8_mlp=self.fp8,
                    moe_deep_gemm=self.moe_deep_gemm,
                    moe_grouped_gemm=self.moe_grouped_gemm,
                    recompute_moe_gate_up=self.recompute_moe_gate_up,
                    recompute_moe_premute=self.recompute_moe_premute,
                    fp8_dispatched_handle=fp8_dispatched_handle,
                    use_bf16_gemm_weight_grad=not self.fp8_wgrad,
                    use_auto_subbatch=self.use_auto_subbatch,
                    moe_expert_fusion=self.moe_expert_fusion,
                    moe_subbatch_token_num_after_dispatch=self.moe_subbatch_token_num_after_dispatch,
                    moe_subbatch_diag=self.moe_subbatch_diag,
                    dw_p2p_overlap=self.dw_p2p_overlap,
                )

        with profile("combine"):
            hidden_states = self.token_dispatcher._comm_manager.combine(
                hidden_states, combine_overlap_handle
            )

        return hidden_states

    def compute_gate(self, hidden_states):
        if self.expert_model_parallel_size <= 1 and self.sequence_parallel:
            hidden_states = GatherOp.apply(hidden_states)
        return self.gate(hidden_states)

    def dispatch_preprocess(self, args):
        hidden_states, token_probs, token_indices = args
        assert isinstance(self.token_dispatcher, MoEFlexTokenDispatcher)
        hidden_states = self.token_dispatcher.dispatch_preprocess_overlap(
            hidden_states, token_probs, token_indices
        )
        token_probs = self.token_dispatcher._comm_manager.token_probs
        token_indices = self.token_dispatcher._comm_manager.token_indices
        return hidden_states, token_indices, token_probs

    def compute_dispatch(self, args, async_finish=False):
        hidden_states, token_indices, token_weights = args
        if self.moe_use_fusion_node:
            dispatched_hidden_states, fp8_dispatched_handle = (
                self.token_dispatcher.token_dispatch_overlap(
                    hidden_states,
                    token_indices,
                    token_weights,
                    self.fp8_dispatch,
                    async_finish=async_finish,
                )
            )
            dispatched_indices = (
                self.token_dispatcher._comm_manager.dispatched_indices
            )
            dispatched_probs = (
                self.token_dispatcher._comm_manager.dispatched_probs
            )
            # NOTE: tokens_per_expert_list is stateful and should be saved for recompute.
            tokens_per_expert = (
                self.token_dispatcher._comm_manager.tokens_per_expert
            )
            # dispatched_hidden_states's dtype is fp8, but its gradient's dtype is bf16, so type separation is required; the actual values are passed via a dictionary.
            dispatched_hidden_states, guard_status = GradDtypeGuard.apply(
                dispatched_hidden_states, hidden_states.dtype
            )
            guard_status["x"].stop_gradient = True

            return (
                dispatched_hidden_states,
                dispatched_indices,
                dispatched_probs,
                fp8_dispatched_handle,
                tokens_per_expert,
                guard_status,
            )

    def compute_experts(self, args, is_first_fwd=False):
        if self.moe_use_fusion_node:
            (
                dispatched_hidden_states,
                dispatched_indices,
                dispatched_probs,
                fp8_dispatched_handle,
                tokens_per_expert,
                guard_status,
            ) = args
            self.token_dispatcher._comm_manager.tokens_per_expert = (
                tokens_per_expert
            )
            dispatched_hidden_states = GradDtypeUnguard.apply(
                dispatched_hidden_states, guard_status
            )
            hidden_states = FusionMoePyLayer.apply(
                dispatched_hidden_states,
                dispatched_probs,
                dispatched_indices.clone()
                if is_first_fwd
                else dispatched_indices,
                self,
                self.num_experts_per_tok,
                use_fp8_mlp=self.fp8,
                moe_deep_gemm=self.moe_deep_gemm,
                moe_grouped_gemm=self.moe_grouped_gemm,
                recompute_moe_gate_up=self.recompute_moe_gate_up,
                recompute_moe_premute=self.recompute_moe_premute,
                fp8_dispatched_handle=fp8_dispatched_handle,
                use_bf16_gemm_weight_grad=not self.fp8_wgrad,
                use_auto_subbatch=self.use_auto_subbatch,
                moe_expert_fusion=self.moe_expert_fusion,
                moe_subbatch_diag=self.moe_subbatch_diag,
                dw_p2p_overlap=self.dw_p2p_overlap,
            )
            if is_first_fwd:
                hidden_states.stop_gradient = False
        else:
            hidden_states, topk_weights = args
            hidden_states = self.routed_experts_compute(hidden_states)
        return hidden_states

    def compute_combine(self, hidden_states, async_finish=False):
        if self.moe_use_fusion_node:
            hidden_states = self.token_dispatcher._comm_manager.combine(
                hidden_states, None, async_finish=async_finish
            )
        else:
            hidden_states = self.combine(hidden_states)
        return hidden_states

    def aux_loss_compute(self, args):
        hidden_states, aux_loss, residuals = args
        if self.training and self.router_aux_loss_coef:
            aux_loss = aux_loss * float(self.router_aux_loss_coef)
            output = AddAuxiliaryLoss.apply(hidden_states, aux_loss)
        else:
            output = hidden_states
        output = output.reshape(residuals.shape)
        if self.shared_experts is not None:
            shared_output = self.shared_experts(residuals)[0]
            output = output + shared_output

        if self.expert_model_parallel_size <= 1 and self.sequence_parallel:
            output = ScatterOp.apply(output)
        return output

    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        """
        Args:
            hidden_states: Shape: [batch_size, seq_len, hidden_size]

        Returns:
            output: Shape: [batch_size, seq_len, hidden_size]
        """
        if self.expert_model_parallel_size <= 1 and self.sequence_parallel:
            hidden_states = GatherOp.apply(hidden_states)
        orig_shape = hidden_states.shape
        residuals = hidden_states

        layer_idx = getattr(self, "layer_number", None)
        _log_moe_md5(hidden_states, "moe_input", layer_idx)

        # Reset debug-dump buffers at the start of each forward so every
        # step's snapshot is independent.
        self._dbg_reset()
        self._dbg_dump("moe_input", hidden_states)

        (
            capacity,
            topk_weights,
            topk_indices,
            gates_masked,
            mask,
            priorities,
            aux_loss,
            z_loss,
        ) = self.gate(hidden_states)
        # topk_weights, topk_indices: Shape is [seq_len, moe_router_topk]
        # gates_masked, mask: Shape is [seq_len, num_experts], sometimes their names are "probs" and "routing_map"
        # capacity, priorities are used for dropping tokens, currently they are not used

        _log_moe_md5(gates_masked, "gates_masked", layer_idx)
        _log_moe_md5(mask, "routing_mask", layer_idx)

        self._dbg_dump("gates_masked", gates_masked)
        self._dbg_dump("topk_weights", topk_weights)
        self._dbg_dump("mask", mask, grad_hook=False)
        self._dbg_dump("topk_indices", topk_indices, grad_hook=False)

        if (
            self.shared_experts is not None
            and self.moe_shared_expert_overlap
            and self.moe_use_fusion_node
            and self.expert_model_parallel_size > 1
        ):
            combine_overlap_handle = {
                "fn": self.shared_experts,
                "fn_args": (residuals,),
            }
        else:
            combine_overlap_handle = None
        if self.expert_model_parallel_size > 1:
            if self.moe_use_fusion_node:
                output = self.fusion_moe_forward(
                    hidden_states, gates_masked, mask, combine_overlap_handle
                )
            else:
                output = self.custom_forward(hidden_states, gates_masked, mask)
        else:
            if len(hidden_states.shape) == 3:
                batch_size, seq_len, d_model = hidden_states.shape
                reshaped_input = hidden_states.reshape([-1, d_model])
            else:
                reshaped_input = hidden_states
            if self.moe_grouped_gemm:
                output = self._forward_single_card_grouped_gemm_moe(
                    reshaped_input,
                    mask,
                    gates_masked,
                    topk_indices,
                    topk_weights,
                )
            else:
                output = self._forward_single_card_moe(
                    reshaped_input, topk_indices, topk_weights
                )

        _log_moe_md5(output, "moe_routed_output", layer_idx)
        self._dbg_dump("routed_output", output)

        if self.training and self.router_aux_loss_coef:
            aux_loss = aux_loss * float(self.router_aux_loss_coef)
            output = AddAuxiliaryLoss.apply(output, aux_loss)

        output = output.reshape(orig_shape)
        if self.shared_experts is not None:
            if combine_overlap_handle is not None:
                shared_output = combine_overlap_handle["fn_out"][0]
            else:
                shared_output = self.shared_experts(residuals)[0]
            output = output + shared_output

        _log_moe_md5(output, "moe_final_output", layer_idx)
        self._dbg_dump("moe_output", output)

        if self.expert_model_parallel_size <= 1 and self.sequence_parallel:
            output = ScatterOp.apply(output)
        return output, None  # None is bias

    def _forward_single_card_moe(
        self,
        hidden_states: paddle.Tensor,
        selected_experts: paddle.Tensor,
        topk_weights: paddle.Tensor,
    ) -> paddle.Tensor:
        """
        Forward without expert parallelism

        Args:
            hidden_states: Input hidden states, shape: [batch_size*seq_len, hidden_size]
            selected_experts: TopK experts indices, shape: [seq_len, num_experts_per_tok]
            topk_weights: TopK weights, shape: [seq_len, num_experts_per_tok]

        Returns:
            output: Output hidden states, shape: [seq_len, hidden_size]
        """

        _, d_model = hidden_states.shape
        final_hidden_states = paddle.zeros_like(
            hidden_states, dtype=hidden_states.dtype
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = paddle.nn.functional.one_hot(
            selected_experts, num_classes=self.num_experts
        ).transpose([2, 1, 0])
        tokens_per_expert = expert_mask.reshape([expert_mask.shape[0], -1]).sum(
            axis=-1
        )
        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            top_x, idx = paddle.where(expert_mask[expert_idx])
            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            if tokens_per_expert[expert_idx] <= 0.1:
                continue
            current_state = hidden_states[idx, None].reshape([-1, d_model])
            expert_out = expert_layer(current_state)[0]
            current_weight = topk_weights[idx, top_x].unsqueeze(-1)
            current_hidden_states = expert_out * current_weight

            # use scatter to replace index_add
            final_hidden_states_tmp = paddle.zeros_like(final_hidden_states)
            final_hidden_states_tmp = paddle.scatter(
                final_hidden_states_tmp,
                idx.reshape([-1]),
                current_hidden_states.to(hidden_states.dtype),
                overwrite=False,
            )
            final_hidden_states = final_hidden_states + final_hidden_states_tmp
        return final_hidden_states.cast(hidden_states.dtype)

    def run_sonic_moe(
        self, hidden_states, topk_indices, topk_scores, fp8=False
    ):
        T = hidden_states.shape[0]
        K = self.num_experts_per_tok
        E = self.num_experts_per_device
        stream_id = paddle.device.cuda.current_stream().cuda_stream

        valid = topk_indices >= 0
        valid_experts = topk_indices[valid].cast(paddle.int32)
        tokens_per_expert = paddle.bincount(valid_experts, minlength=E).cast(
            paddle.int32
        )

        (
            expert_frequency_offset,
            x_gather_idx,
            s_scatter_idx,
            s_reverse_scatter_idx,
            num_activated_expert_per_token_offset,
            _router_scores,
            TK_padded,
            total_pad_rows,
            _N_recv,
            _score_src_idx,
        ) = deepep_topk_to_sonic_metadata(
            topk_indices.cast(paddle.int32),
            topk_scores,
            tokens_per_expert,
            E,
            block=128 if fp8 else 1,
        )

        s_scatter_idx.stop_gradient = True
        w1 = self.grouped_gemm_experts.weight1
        w2 = self.grouped_gemm_experts.weight2
        activation_type = ActivationType("swiglu")

        total_expert_freq = TK_padded
        scores_for_down = _differentiable_router_scores(
            topk_scores,
            topk_indices.cast(paddle.int32),
            num_activated_expert_per_token_offset,
            TK_padded - total_pad_rows,
            TK_padded,
            E,
            score_src_idx=_score_src_idx,
        )

        print("==== fp8:", fp8)
        # FP8 path: use sonicmoe's optimized FP8 GEMM kernels.
        # Scores are applied AFTER the down-projection inside
        # _router_forward — mathematically equivalent to the
        # baseline's score-before-downproj for per-row scalars.
        # if fp8:

        with enable_fp8(fp8):
            _refresh_fp8_config()
            y1, z = _UpProjection.apply(
                hidden_states,
                w1.permute([1, 2, 0]),
                None,
                expert_frequency_offset,
                total_expert_freq,
                K,
                stream_id,
                x_gather_idx,
                s_scatter_idx,
                s_reverse_scatter_idx,
                num_activated_expert_per_token_offset,
                True,  # is_varlen_k
                activation_type,
                is_inference_mode_enabled=False,
                use_low_precision_postact_buffer=False,
            )
            self._dbg_dump("sonic_y1", y1)
            # `z` is the pre-SwiGLU fc1 output.  Sonic stores it in
            # interleaved layout [g0, u0, g1, u1, ...] along the last
            # axis, whereas baseline GroupedMLPExpert produces it in
            # concat layout [gate_all | up_all].  De-interleave here so
            # the dumped tensor matches the baseline layout and can be
            # diffed element-wise.
            #
            # The forward snapshot is taken from a detached clone of z
            # (autograd-free; the reshape/concat is cheap).  The
            # backward hook must be registered on the *original* z
            # (which is actually in the graph via _DownProjection),
            # and the interleaved grad is reshaped to concat form
            # inside the callback before storing — a hook registered
            # on the detached/reshaped tensor would never fire.
            if self._debug_dump_enabled and z is not None:
                try:
                    dbg_z = z.detach().clone()
                    two_i = dbg_z.shape[-1]
                    z_pair = dbg_z.reshape([*dbg_z.shape[:-1], two_i // 2, 2])
                    z_concat = paddle.concat(
                        [z_pair[..., 0], z_pair[..., 1]], axis=-1
                    )
                    self._dbg_dump("sonic_z", z_concat, grad_hook=False)
                    if not getattr(z, "stop_gradient", True):
                        store = self._debug_dump["bwd"]

                        def _z_hook(grad, s=store):
                            try:
                                ti = grad.shape[-1]
                                gpair = grad.reshape(
                                    [*grad.shape[:-1], ti // 2, 2]
                                )
                                g_concat = paddle.concat(
                                    [gpair[..., 0], gpair[..., 1]], axis=-1
                                )
                                s["sonic_z"] = g_concat.detach().clone()
                            except Exception:
                                s["sonic_z"] = grad.detach().clone()

                        try:
                            z.register_hook(_z_hook)
                        except Exception:
                            pass
                except Exception:
                    self._dbg_dump("sonic_z", z)

            hidden_states = _DownProjection.apply(
                y1,
                z,
                w2.permute([1, 2, 0]),
                None,
                scores_for_down,
                s_scatter_idx,
                expert_frequency_offset,
                T,
                K,
                stream_id,
                x_gather_idx,
                s_scatter_idx,
                s_reverse_scatter_idx,
                num_activated_expert_per_token_offset,
                True,  # is_varlen_k
                activation_type,
                None,
            )
            self._dbg_dump("sonic_down_out", hidden_states)

        return hidden_states

    def _forward_single_card_grouped_gemm_moe(
        self,
        hidden_states: paddle.Tensor,
        routing_map: paddle.Tensor,
        probs: paddle.Tensor,
        topk_indices: paddle.Tensor | None = None,
        topk_scores: paddle.Tensor | None = None,
    ) -> paddle.Tensor:
        """
        Forward without expert parallelism

        Args:
            hidden_states: Input hidden states, shape: [batch_size*seq_len, hidden_size]
            routing_map: Routing map, shape: [seq_len, num_experts]
            probs: Probabilities of selecting each expert, shape: [seq_len, num_experts]

        Returns:
            output: Output hidden states, shape: [seq_len, hidden_size]
        """

        def _convert_routing_map_and_probs(
            routing_map: paddle.Tensor, probs: paddle.Tensor, topk: int
        ):
            routing_map = routing_map.astype("bool")
            masked_probs = probs * routing_map.astype("float32")
            weights, indices = paddle.topk(masked_probs, k=topk, axis=-1)
            return indices, weights

        if self.using_sonic_moe:
            use_fp8 = self.fp8 is not None
            final_hidden_states = self.run_sonic_moe(
                hidden_states, topk_indices, topk_scores, use_fp8
            )
            return final_hidden_states.cast(hidden_states.dtype)
        else:
            tokens_per_expert = routing_map.sum(axis=0)
            permuted_local_hidden_states, sorted_indices = permute(
                hidden_states, routing_map, tokens_per_expert
            )
            self._dbg_dump(
                "baseline_permuted_input", permuted_local_hidden_states
            )
            grouped_expert_out = self.grouped_gemm_experts(
                permuted_local_hidden_states, tokens_per_expert
            )[0]
            final_hidden_states = unpermute(
                grouped_expert_out,
                sorted_indices,
                restore_shape=hidden_states.shape,
                probs=probs,
                routing_map=routing_map,
            )
            # sonic's `_DownProjection` returns a post-scatter+combine
            # tensor of shape [T, H]; baseline's equivalent is the
            # output of `unpermute` (which also weights and combines
            # the k expert contributions).  Dump under sonic key name
            # so diff tool aligns them directly.
            self._dbg_dump("sonic_down_out", final_hidden_states)
            return final_hidden_states.cast(hidden_states.dtype)

    def fp8_quant_weight(self, batch_mode=False, quant_transpose=True):
        if not (self.moe_use_fusion_node and self.fp8):
            return

        def quantize_weights(
            weight_list, weight_obj=None, quant_transpose=None
        ):
            """Helper function to quantize a list of weights."""
            if weight_obj is None:
                weight_obj = weight_list[0]

            if quant_transpose is None:
                fp8_weight, fp8_scale = fused_stack_quant_without_cache(
                    weight_list, transpose=False
                )
                weight_obj.fp8_weight_stacked = fp8_weight
                weight_obj.fp8_scale_stacked = fp8_scale

                fp8_weight_t, fp8_scale_t = fused_stack_quant_without_cache(
                    weight_list, transpose=True
                )
                weight_obj.fp8_weight_stacked_transpose = fp8_weight_t
                weight_obj.fp8_scale_stacked_transpose = fp8_scale_t
            elif quant_transpose is False:
                # Only quantize without transpose
                fp8_weight, fp8_scale = fused_stack_quant_without_cache(
                    weight_list, transpose=False
                )
                weight_obj.fp8_weight_stacked = fp8_weight
                weight_obj.fp8_scale_stacked = fp8_scale
            elif quant_transpose is True:
                # Only quantize with transpose
                fp8_weight_t, fp8_scale_t = fused_stack_quant_without_cache(
                    weight_list, transpose=True
                )
                weight_obj.fp8_weight_stacked_transpose = fp8_weight_t
                weight_obj.fp8_scale_stacked_transpose = fp8_scale_t
            else:
                raise ValueError("Invalid value for `quant_transpose`.")

        if hasattr(self, "grouped_gemm_experts"):
            if batch_mode:
                expert_w1 = self.grouped_gemm_experts.weight1
                expert_w2 = self.grouped_gemm_experts.weight2
                local_expert_num = expert_w1.shape[0]
                expert_w1_list = [
                    expert_w1[i, :, :] for i in range(local_expert_num)
                ]
                expert_w2_list = [
                    expert_w2[i, :, :] for i in range(local_expert_num)
                ]

                # Batch mode: process all experts' weights together
                if expert_w1_list:
                    quantize_weights(
                        expert_w1_list,
                        self.grouped_gemm_experts.weight1,
                        quant_transpose,
                    )
                if expert_w2_list:
                    quantize_weights(
                        expert_w2_list,
                        self.grouped_gemm_experts.weight2,
                        quant_transpose,
                    )

            else:
                raise NotImplementedError(
                    "Not support individual mode for fuse_expert_fp8_weight_quant yet."
                )

            return

        if batch_mode:
            # Batch mode: process all experts' weights together
            expert_w1_list = [
                expert.up_gate_proj.weight
                for expert in self.experts
                if expert is not None
            ]
            expert_w2_list = [
                expert.down_proj.weight
                for expert in self.experts
                if expert is not None
            ]
            if expert_w1_list:
                quantize_weights(
                    expert_w1_list, expert_w1_list[0], quant_transpose
                )
            if expert_w2_list:
                quantize_weights(
                    expert_w2_list, expert_w2_list[0], quant_transpose
                )

        else:
            # Individual mode: process each expert's weights separately
            for expert in self.experts:
                if expert is not None:
                    quantize_weights(
                        [expert.up_gate_proj.weight],
                        quant_transpose=quant_transpose,
                    )
                    quantize_weights(
                        [expert.down_proj.weight],
                        quant_transpose=quant_transpose,
                    )

    def use_fp8(self):
        if self.moe_use_fusion_node and self.fp8:
            return True
        return False

    def set_layer_number(self, layer_number):
        self.layer_number = layer_number
        assert hasattr(self.gate, "set_layer_number"), (
            "expect gate has method 'set_layer_number'"
        )
        self.gate.set_layer_number(layer_number)
