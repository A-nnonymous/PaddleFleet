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


import os
import random
import subprocess
import sys
import unittest

# Enable intermediate-tensor dumps in MoELayer before importing it so
# the module-level env check flips on for the whole test process.
os.environ.setdefault("MOE_DEBUG_DUMP", "1")

import numpy as np
import paddle

paddle.compat.enable_torch_proxy(
    scope={"sonicmoe", "paddlefleet.ops.sonicmoe", "quack", "triton"},
    silent=True,
)
import paddle.nn.functional as F
from paddle.distributed import fleet

# from paddlefleet.tensor_parallel.random import model_parallel_cuda_manual_seed
from paddle.distributed.fleet.utils import mix_precision_utils

import paddlefleet

# from tests.unit_tests.test_utilities import Utils
import paddlefleet.parallel_state as ps
from paddlefleet.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from paddlefleet.ops.utils import get_cuda_version
from paddlefleet.process_groups_config import ProcessGroupCollection
from paddlefleet.transformer.moe.moe_layer import MoELayer
from paddlefleet.transformer.transformer_config import TransformerConfig

if paddlefleet.ops.is_sonic_moe_available():
    from paddlefleet.ops.sonicmoe.functional import (
        clear_all_fp8_weight_caches,
    )

# ── Prevent duplicate custom-op registration ──────────────────────────
# paddlefleet bundles a copy of sonicmoe at paddlefleet.ops.sonicmoe,
# while the standalone sonicmoe is also editable-installed.  Paddle's
# import proxy can trigger a fresh import of the standalone package
# during backward, causing "sonicmoe::count_cumsum_cuda already
# registered".  Alias the bundled modules into the top-level namespace
# so the import machinery finds them in sys.modules and skips re-exec.
for _key in list(sys.modules):
    if _key.startswith("paddlefleet.ops.sonicmoe"):
        _alias = _key.replace("paddlefleet.ops.sonicmoe", "sonicmoe", 1)
        sys.modules.setdefault(_alias, sys.modules[_key])


def get_gpu_models_via_nvidia_smi():
    try:
        output = subprocess.check_output(
            "nvidia-smi --query-gpu=name --format=csv,noheader", shell=True
        )
        models = output.decode().strip().replace("NVIDIA", "")
        return models
    except Exception as e:
        return ["Unknown"]


def judge_machine_type():
    if not paddle.is_compiled_with_cuda():
        return "No CUDA GPU"
    models = get_gpu_models_via_nvidia_smi()
    for model in models:
        name = model.upper()
        if "V" in name:
            return "V"
        elif "H" in name:
            return "H"


result = judge_machine_type()
print("你的机器类型是：", result)
version, cuda_minor = get_cuda_version()
print("CUDA version:", version)


def calc_diff(x: paddle.Tensor, y: paddle.Tensor):
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    if denominator.item() == 0:
        return 0.0
    sim = 2 * (x * y).sum() / denominator
    return (1 - sim).item()


# ── Module-level fleet initialization (only once) ─────────────────────
_strategy = fleet.DistributedStrategy()
_strategy.hybrid_configs = {
    "dp_degree": 1,
    "mp_degree": 1,
    "pp_degree": 1,
    "sharding_degree": 1,
    "sep_degree": 1,
    "cp_degree": 1,
    "ep_degree": 1,
    "moe_sharding_degree": 1,
    "order": [
        "sharding",
        "moe_sharding",
        "pp",
        "sep",
        "cp",
        "dp",
        "ep",
        "mp",
    ],
}
fleet.init(is_collective=True, strategy=_strategy)
_hcg = fleet.get_hybrid_communicate_group()
ps.initialize_model_parallel(_hcg)


# @unittest.skipUnless(
#     paddlefleet.ops.is_sonic_moe_available(),
#     "Sonic-MoE not available (requires Python>=3.12, CUDA>=12.9, SM>=90)",
# )
# class TestSonicMoEPrecision(unittest.TestCase):
#     """Precision comparison: baseline grouped_gemm vs BF16 sonic-moe vs FP8 sonic-moe."""

#     def setUp(self):
#         self.strategy = _strategy
#         self.acc_step = 4

#     def _base_config_kwargs(self):
#         """Common config kwargs shared by all variants."""
#         return {
#             "num_hidden_layers": 2,
#             "hidden_size": 512,
#             "vocab_size": 100,
#             "max_sequence_length": 64,
#             "num_attention_heads": 4,
#             "intermediate_size": 1024,
#             "normalization": "RMSNorm",
#             "hidden_dropout_prob": 0.0,
#             "attention_dropout": 0.0,
#             "n_routed_experts": 8,
#             "use_bias": False,
#             "rotary_percent": 1.0,
#             "rotary_base": 10000,
#             "rope_scaling": 1.0,
#             "moe_intermediate_size": 1024,
#             "moe_token_dispatcher_type": "alltoall",
#             "n_shared_experts": 1,
#             "init_method": functools.partial(
#                 paddle.nn.init.xavier_uniform_, gain=1.0
#             ),
#             "output_layer_init_method": functools.partial(
#                 paddle.nn.init.xavier_uniform_, gain=1.0
#             ),
#             "use_qk_norm": True,
#             # All variants use gated_linear_unit=True so that the baseline
#             # GroupedMLPExpert also computes SwiGLU, matching sonic-moe's
#             # internal activation.  This ensures a fair comparison.
#             "gated_linear_unit": True,
#         }

#     def _build_model(self, seed=46, **extra):
#         """Build a GPT MoE model with deterministic initialization."""
#         random.seed(seed)
#         np.random.seed(seed)
#         paddle.seed(seed)
#         kw = self._base_config_kwargs()
#         kw.update(extra)
#         config = GPTConfig(**kw)
#         model = gpt_builder(config, num_stages=1)
#         return config, model

#     def _make_data(self, config, step_idx=0, micro_batch_size=2):
#         """Create deterministic input data for one accumulation step."""
#         seq = config.max_sequence_length
#         token_offset = step_idx * seq
#         ids = [
#             (token_offset + idx) % config.vocab_size for idx in range(seq)
#         ]
#         labels = [
#             (token_offset + idx + 1) % config.vocab_size for idx in range(seq)
#         ]
#         input_ids = paddle.to_tensor(ids, dtype=paddle.int64).repeat(
#             (micro_batch_size, 1)
#         )
#         position_ids = paddle.to_tensor(list(range(seq)), dtype=paddle.int64).repeat(
#             (micro_batch_size, 1)
#         )
#         attention_mask = paddle.ones(
#             (micro_batch_size, 1, seq, seq), dtype=bool
#         )
#         labels = paddle.to_tensor(labels, dtype=paddle.int64).repeat(
#             (micro_batch_size, 1)
#         )
#         return (
#             {
#                 "input_ids": [input_ids],
#                 "position_ids": [position_ids],
#                 "attention_mask": [attention_mask],
#             },
#             [labels],
#         )

#     def _make_accumulation_data(self, config):
#         return [
#             self._make_data(config, step_idx=step_idx)
#             for step_idx in range(self.acc_step)
#         ]

#     def _forward_backward(self, model, data_steps):
#         """Run accumulated forward + backward and return ([loss_values], grads)."""
#         pipe = NoPipelineParallel(model, self.strategy)
#         pipe = paddle.amp.decorate(
#             models=pipe,
#             level="O2",
#             dtype="bfloat16",
#             master_grad=True,
#             master_weight=True,
#         )
#         mix_precision_utils.MixPrecisionLayer(pipe, dtype="bfloat16")

#         losses = []
#         for data in data_steps:
#             with paddle.amp.auto_cast(level="O2", dtype="bfloat16"):
#                 loss = pipe.forward_backward_pipeline(data)
#             losses.append(loss.item())

#         grads = {}
#         for name, p in pipe.named_parameters():
#             grad = getattr(p, "main_grad", None)
#             if grad is not None:
#                 grads[name] = grad.detach().clone()
#         return losses, grads

#     @staticmethod
#     def _split_to_sonic_interleaved(weight):
#         gate, up = paddle.chunk(weight, 2, axis=-1)
#         gate = gate.transpose([0, 2, 1])
#         up = up.transpose([0, 2, 1])
#         return paddle.stack([gate, up], axis=2).reshape(
#             weight.shape[0], -1, weight.shape[1]
#         )

#     @staticmethod
#     def _sonic_interleaved_to_split_grad(grad):
#         grad = grad.reshape([grad.shape[0], -1, 2, grad.shape[2]])
#         gate = grad[:, :, 0, :].transpose([0, 2, 1])
#         up = grad[:, :, 1, :].transpose([0, 2, 1])
#         return paddle.concat([gate, up], axis=-1)

#     @classmethod
#     def _aligned_grad_for_compare(
#         cls, name, grad, transpose_grouped_gemm=False
#     ):
#         if not transpose_grouped_gemm:
#             return grad
#         if "grouped_gemm_experts.weight1" in name:
#             return cls._sonic_interleaved_to_split_grad(grad)
#         if "grouped_gemm_experts.weight2" in name:
#             return grad.transpose([0, 2, 1])
#         return grad

#     @classmethod
#     def _copy_weights_to_sonic(cls, src_model, dst_model):
#         """Copy weights from a non-sonic-moe model to a sonic-moe model.

#         GroupedMLPExpert weight layouts differ by using_sonic_moe:
#           non-sonic: w1=[E, hidden, fc1_out], w2=[E, fc2_in, hidden]
#           sonic:     w1=[E, fc1_out, hidden], w2=[E, hidden, fc2_in]
#         Sonic-MoE stores w1 rows interleaved as gate/up pairs, so w1 is
#         transposed and interleaved. w2 is only transposed on axes [1, 2].
#         All other weights (embedding, attention, norm, shared expert) share
#         the same shape and are copied directly.
#         """
#         src_params = dict(src_model.named_parameters())
#         for name, dst_p in dst_model.named_parameters():
#             src_p = src_params[name]
#             if "grouped_gemm_experts.weight1" in name:
#                 dst_p.set_value(cls._split_to_sonic_interleaved(src_p))
#             elif "grouped_gemm_experts.weight2" in name:
#                 dst_p.set_value(src_p.transpose([0, 2, 1]))
#             else:
#                 dst_p.set_value(src_p.clone())

#     def test_precision_comparison(self) -> None:
#         # ── 1. Baseline: grouped_gemm BF16 (no sonic-moe) ──────────────
#         cfg_base, model_base = self._build_model(
#             moe_grouped_gemm=True,
#             using_sonic_moe=False,
#         )

#         # ── 2. BF16 sonic-moe ──────────────────────────────────────────
#         cfg_bf16, model_bf16 = self._build_model(
#             moe_grouped_gemm=True,
#             using_sonic_moe=True,
#         )
#         self._copy_weights_to_sonic(model_base, model_bf16)

#         # ── 3. FP8 sonic-moe ──────────────────────────────────────────
#         cfg_fp8, model_fp8 = self._build_model(
#             moe_grouped_gemm=True,
#             using_sonic_moe=True,
#             fp8="e4m3",
#         )
#         self._copy_weights_to_sonic(model_base, model_fp8)

#         # ── Run accumulated forward + backward for each variant ────────
#         loss_base_steps, grads_base = self._forward_backward(
#             model_base, self._make_accumulation_data(cfg_base)
#         )
#         print(f"[Baseline] accumulated losses = {loss_base_steps}")

#         loss_bf16_steps, grads_bf16 = self._forward_backward(
#             model_bf16, self._make_accumulation_data(cfg_bf16)
#         )
#         print(f"[BF16 sonic-moe] accumulated losses = {loss_bf16_steps}")

#         loss_fp8_steps, grads_fp8 = self._forward_backward(
#             model_fp8, self._make_accumulation_data(cfg_fp8)
#         )
#         clear_all_fp8_weight_caches()
#         print(f"[FP8  sonic-moe] accumulated losses = {loss_fp8_steps}")

#         self.assertEqual(len(loss_base_steps), self.acc_step)
#         self.assertEqual(len(loss_bf16_steps), self.acc_step)
#         self.assertEqual(len(loss_fp8_steps), self.acc_step)

#         # ── Check BF16 sonic-moe vs Baseline for each accumulation step ─
#         for step_idx, (loss_base, loss_bf16) in enumerate(
#             zip(loss_base_steps, loss_bf16_steps), start=1
#         ):
#             rdiff_bf16 = abs(loss_bf16 - loss_base) / abs(loss_base)
#             print(
#                 f"BF16 vs Baseline [acc step {step_idx}/{self.acc_step}]: "
#                 f"loss relative diff = {rdiff_bf16:.6e}"
#             )
#             self.assertLess(
#                 rdiff_bf16,
#                 1e-7,
#                 f"BF16 sonic-moe loss deviates too much from baseline at "
#                 f"acc_step={step_idx} (baseline={loss_base}, bf16={loss_bf16})",
#             )

#         common_bf16_grads = set(grads_base) & set(grads_bf16)
#         self.assertTrue(
#             common_bf16_grads, "No common BF16 accumulated main_grad tensors found"
#         )
#         for name in sorted(common_bf16_grads):
#             g0 = grads_base[name]
#             g1 = self._aligned_grad_for_compare(
#                 name, grads_bf16[name], transpose_grouped_gemm=True
#             )
#             diff = calc_diff(g0, g1)
#             print(
#                 f"BF16 vs Baseline accumulated grad diff = {diff:.6e} for {name}"
#             )
#             self.assertLess(
#                 diff,
#                 1e-5,
#                 f"BF16 accumulated grad tensor diff too large for {name}: "
#                 f"diff={diff:.6e}",
#             )

#         # ── Check FP8 sonic-moe vs BF16 sonic-moe for each acc step ────
#         for step_idx, (loss_bf16, loss_fp8) in enumerate(
#             zip(loss_bf16_steps, loss_fp8_steps), start=1
#         ):
#             rdiff_fp8 = abs(loss_fp8 - loss_bf16) / abs(loss_bf16)
#             print(
#                 f"FP8 vs BF16 [acc step {step_idx}/{self.acc_step}]: "
#                 f"loss relative diff = {rdiff_fp8:.6e}"
#             )
#             self.assertLess(
#                 rdiff_fp8,
#                 1e-6,
#                 f"FP8 sonic-moe loss deviates too much from BF16 at "
#                 f"acc_step={step_idx} (bf16={loss_bf16}, fp8={loss_fp8})",
#             )

#         common_fp8_grads = set(grads_bf16) & set(grads_fp8)
#         self.assertTrue(
#             common_fp8_grads, "No common FP8 accumulated main_grad tensors found"
#         )
#         for name in sorted(common_fp8_grads):
#             g1 = grads_bf16[name]
#             g2 = grads_fp8[name]
#             diff = calc_diff(g1, g2)
#             grad_tol = 3e-3
#             print(f"FP8 vs BF16 accumulated grad diff = {diff:.6e} for {name}")
#             self.assertLess(
#                 diff,
#                 grad_tol,
#                 f"FP8 accumulated grad tensor diff too large for {name}: "
#                 f"diff={diff:.6e}, tol={grad_tol}",
#             )

#         print("All accumulation precision comparison checks passed!")


@unittest.skipUnless(
    paddlefleet.ops.is_sonic_moe_available(),
    "Sonic-MoE not available (requires Python>=3.12, CUDA>=12.9, SM>=90)",
)
class TestSonicMoELayerPrecision(unittest.TestCase):
    """Precision comparison at the MoELayer level:
    baseline grouped_gemm vs BF16 sonic-moe vs FP8 sonic-moe.
    """

    def setUp(self):
        self.pg_collection = ProcessGroupCollection.use_mpu_process_groups()

        self.seed = 46
        self.hidden_size = 256
        self.n_routed_experts = 8
        self.acc_steps = 1

    def _build_transformer_config(self, using_sonic_moe=False, fp8=None):
        return TransformerConfig(
            hidden_size=self.hidden_size,
            num_attention_heads=4,
            n_routed_experts=self.n_routed_experts,
            use_cpu_initialization=False,
            num_experts_per_tok=2,
            tensor_model_parallel_size=1,
            expert_model_parallel_size=1,
            sequence_parallel=False,
            bf16=True,
            params_dtype=paddle.bfloat16,
            moe_intermediate_size=512,
            gated_linear_unit=True,
            n_shared_experts=0,
            hidden_act=F.silu,
            moe_grouped_gemm=True,
            bias_activation_fusion=True,
            moe_token_dispatcher_type="alltoall",
            moe_use_fusion_node=True,
            using_sonic_moe=using_sonic_moe,
            fp8=fp8,
            fp8_wgrad=True,
        )

    def _build_moe_layer(self, using_sonic_moe=False, fp8=None):
        random.seed(self.seed)
        np.random.seed(self.seed)
        paddle.seed(self.seed)
        transformer_config = self._build_transformer_config(
            using_sonic_moe=using_sonic_moe, fp8=fp8
        )
        transformer_layer_spec = get_gpt_layer_local_spec(
            transformer_config,
            num_experts=self.n_routed_experts,
        )

        moe_layer = MoELayer(
            transformer_config,
            transformer_layer_spec.sublayers_spec.mlp.extra_kwargs["sublayers"],
            self.pg_collection,
        )
        # pipe = NoPipelineParallel(model, self.strategy)
        amp_moe_layer = paddle.amp.decorate(
            models=moe_layer,
            level="O2",
            dtype="bfloat16",
            master_grad=True,
            master_weight=True,
        )
        mix_precision_utils.MixPrecisionLayer(amp_moe_layer, dtype="bfloat16")

        return amp_moe_layer, moe_layer

    @staticmethod
    def _split_to_sonic_interleaved(weight):
        gate, up = paddle.chunk(weight, 2, axis=-1)
        gate = gate.transpose([0, 2, 1])
        up = up.transpose([0, 2, 1])
        return paddle.stack([gate, up], axis=2).reshape(
            [weight.shape[0], -1, weight.shape[1]]
        )

    @staticmethod
    def _sonic_interleaved_to_split_grad(grad):
        grad = grad.reshape([grad.shape[0], -1, 2, grad.shape[2]])
        gate = grad[:, :, 0, :].transpose([0, 2, 1])
        up = grad[:, :, 1, :].transpose([0, 2, 1])
        return paddle.concat([gate, up], axis=-1)

    @classmethod
    def _aligned_grad_for_compare(
        cls, name, grad, transpose_grouped_gemm=False
    ):
        if not transpose_grouped_gemm:
            return grad
        if "grouped_gemm_experts.weight1" in name:
            return cls._sonic_interleaved_to_split_grad(grad)
        if "grouped_gemm_experts.weight2" in name:
            return grad.transpose([0, 2, 1])
        return grad

    @classmethod
    def _copy_weights(cls, src_layer, dst_layer, transpose_grouped_gemm=False):
        src_params = dict(src_layer.named_parameters())
        for name, dst_param in dst_layer.named_parameters():
            src_param = src_params[name]
            if (
                transpose_grouped_gemm
                and "grouped_gemm_experts.weight1" in name
            ):
                dst_param.set_value(cls._split_to_sonic_interleaved(src_param))
            elif (
                transpose_grouped_gemm
                and "grouped_gemm_experts.weight2" in name
            ):
                dst_param.set_value(src_param.transpose([0, 2, 1]))
            else:
                dst_param.set_value(src_param.clone())

    @staticmethod
    def _collect_grads(layer):
        grads = {}
        for name, param in layer.named_parameters():
            grad = getattr(param, "main_grad", None)
            if grad is None:
                grad = param.grad
            if grad is not None:
                grads[name] = grad.detach().clone()
        return grads

    @staticmethod
    def _clear_grads(layer):
        for _, param in layer.named_parameters():
            if hasattr(param, "main_grad") and param.main_grad is not None:
                param.main_grad.zero_()
            if param.grad is not None:
                param.grad.zero_()

    def _run_forward_backward(self, moe_layer, input_data):
        """Run single forward + backward and return (output, loss, input_grad, grads)."""
        hidden_states = input_data.detach().clone()
        hidden_states.stop_gradient = False
        with paddle.amp.auto_cast(level="O2", dtype="bfloat16"):
            output = moe_layer(hidden_states)[0]
            loss = output.sum()
        loss.backward()
        return (
            output.detach().clone(),
            loss.item(),
            hidden_states.grad.detach().clone(),
            self._collect_grads(moe_layer),
        )

    def _run_accumulated_forward_backward(
        self, moe_layer, input_data_list, inner_moe_layer=None
    ):
        """Run multiple forward-backward steps to accumulate gradients.

        If `inner_moe_layer` is provided and its `_debug_dump_enabled` is
        true, the per-step snapshot of forward/backward intermediate
        tensors is collected as well.
        """
        self._clear_grads(moe_layer)
        losses = []
        outputs = []
        dumps = []
        dump_enabled = inner_moe_layer is not None and getattr(
            inner_moe_layer, "_debug_dump_enabled", False
        )
        for input_data in input_data_list:
            hidden_states = input_data.detach().clone()
            hidden_states.stop_gradient = False
            with paddle.amp.auto_cast(level="O2", dtype="bfloat16"):
                output = moe_layer(hidden_states)[0]
                loss = output.sum()
            loss.backward()
            losses.append(loss.item())
            outputs.append(output.detach().clone())
            if dump_enabled:
                # Deep-copy so the next forward's _dbg_reset() does not
                # clobber the snapshot we just took.
                snap = {
                    "fwd": {
                        k: v.detach().clone()
                        for k, v in inner_moe_layer._debug_dump["fwd"].items()
                    },
                    "bwd": {
                        k: v.detach().clone()
                        for k, v in inner_moe_layer._debug_dump["bwd"].items()
                    },
                }
                dumps.append(snap)
        grads = self._collect_grads(moe_layer)
        return losses, outputs, grads, dumps

    @staticmethod
    def _safe_calc_diff(a, b):
        """Diff that tolerates shape/dtype mismatches by falling back to
        float casting and shape-guard.  Returns None when tensors are
        truly incompatible."""
        try:
            if a.shape != b.shape:
                return None
            return calc_diff(a, b)
        except Exception:
            return None

    # Execution order of forward intermediates within MoELayer.  Keys
    # that don't exist on a given path (e.g. baseline_* on sonic, or
    # sonic_* on baseline when GroupedMLPExpert isn't instrumented) are
    # simply skipped during diff printing.
    _FWD_EXEC_ORDER = (
        "moe_input",
        "gates_masked",
        "topk_weights",
        "mask",
        "topk_indices",
        "baseline_permuted_input",
        "sonic_y1",
        "sonic_z",
        "sonic_down_out",
        "baseline_grouped_expert_out",
        "baseline_unpermuted",
        "routed_output",
        "moe_output",
    )

    def _print_intermediate_diffs(
        self,
        label,
        dumps_lhs,
        dumps_rhs,
        lhs_is_sonic=False,
        rhs_is_sonic=False,
    ):
        """Diff forward + backward intermediate tensors step by step.

        Tensors are printed in computation execution order:
        forward follows ``_FWD_EXEC_ORDER``; backward uses the reverse
        order (since gradients flow back through the graph).  Keys that
        exist on only one side are reported as 'skipped (missing on
        <side>)'.  Keys with incompatible shape are reported as
        'skipped (shape mismatch)'.
        """
        n_steps = min(len(dumps_lhs), len(dumps_rhs))
        fwd_order = list(self._FWD_EXEC_ORDER)
        bwd_order = list(reversed(self._FWD_EXEC_ORDER))
        print(f"\n==== {label} intermediate diffs ====")
        for step in range(n_steps):
            lhs = dumps_lhs[step]
            rhs = dumps_rhs[step]
            for side, order in (("fwd", fwd_order), ("bwd", bwd_order)):
                seen = set()
                ordered_keys = [
                    k for k in order if k in lhs[side] or k in rhs[side]
                ]
                seen.update(ordered_keys)
                # Fallback: any keys not listed in exec order go last,
                # sorted alphabetically, so new intermediates aren't
                # silently dropped.
                extras = sorted((set(lhs[side]) | set(rhs[side])) - seen)
                for k in ordered_keys + extras:
                    if k not in lhs[side]:
                        print(
                            f"[{label}][step {step + 1}][{side}/{k}] "
                            f"skipped (missing on lhs)"
                        )
                        continue
                    if k not in rhs[side]:
                        print(
                            f"[{label}][step {step + 1}][{side}/{k}] "
                            f"skipped (missing on rhs)"
                        )
                        continue
                    diff = self._safe_calc_diff(lhs[side][k], rhs[side][k])
                    if diff is None:
                        print(
                            f"[{label}][step {step + 1}][{side}/{k}] "
                            f"skipped (shape/dtype mismatch: "
                            f"lhs={list(lhs[side][k].shape)}, "
                            f"rhs={list(rhs[side][k].shape)})"
                        )
                    else:
                        print(
                            f"[{label}][step {step + 1}][{side}/{k}] "
                            f"diff = {diff:.6e} "
                            f"shape={list(lhs[side][k].shape)}"
                        )

    def test_moe_layer_precision(self):
        """Test MoELayer forward/backward: baseline vs BF16 sonic-moe vs FP8 sonic-moe."""
        # ── 1. Build MoE layers ──────────────────────────────────────────
        moe_layer_base, inner_base = self._build_moe_layer(
            using_sonic_moe=False
        )
        moe_layer_sonic_bf16, inner_bf16 = self._build_moe_layer(
            using_sonic_moe=True
        )
        moe_layer_sonic_fp8, inner_fp8 = self._build_moe_layer(
            using_sonic_moe=True, fp8="e4m3"
        )

        # ── 2. Copy weights from baseline to sonic variants ──────────────
        self._copy_weights(
            moe_layer_base,
            moe_layer_sonic_bf16,
            transpose_grouped_gemm=True,
        )
        self._copy_weights(
            moe_layer_base,
            moe_layer_sonic_fp8,
            transpose_grouped_gemm=True,
        )

        # ── 3. Generate input data for accumulation steps ────────────────
        input_data_list = []
        for step_idx in range(self.acc_steps):
            paddle.seed(self.seed + step_idx)
            data = paddle.randn(
                [2, 64, self.hidden_size], dtype=paddle.bfloat16
            )
            input_data_list.append(data)

        # ── 4. Run accumulated forward-backward ──────────────────────────
        losses_base, outputs_base, grads_base, dumps_base = (
            self._run_accumulated_forward_backward(
                moe_layer_base, input_data_list, inner_base
            )
        )
        print(f"[Baseline] accumulated losses = {losses_base}")

        losses_bf16, outputs_bf16, grads_bf16, dumps_bf16 = (
            self._run_accumulated_forward_backward(
                moe_layer_sonic_bf16, input_data_list, inner_bf16
            )
        )
        print(f"[Sonic-MoE BF16] accumulated losses = {losses_bf16}")

        losses_fp8, outputs_fp8, grads_fp8, dumps_fp8 = (
            self._run_accumulated_forward_backward(
                moe_layer_sonic_fp8, input_data_list, inner_fp8
            )
        )
        clear_all_fp8_weight_caches()
        print(f"[Sonic-MoE FP8] accumulated losses = {losses_fp8}")

        # ── 5. Verify step counts ────────────────────────────────────────
        self.assertEqual(len(losses_base), self.acc_steps)
        self.assertEqual(len(losses_bf16), self.acc_steps)
        self.assertEqual(len(losses_fp8), self.acc_steps)

        # ── 6. BF16 sonic-moe vs Baseline: per-step loss ─────────────────
        bf16_loss_rtol = 1e-2
        bf16_loss_atol = 1e-5
        for step_idx, (loss_base, loss_bf16) in enumerate(
            zip(losses_base, losses_bf16), start=1
        ):
            adiff = abs(loss_bf16 - loss_base)
            rdiff = adiff / max(abs(loss_base), 1e-12)
            print(
                f"BF16 vs Baseline [acc step {step_idx}/"
                f"{self.acc_steps}]: loss relative diff = {rdiff:.6e}"
                f", absolute diff = {adiff:.6e}"
            )
            self.assertTrue(
                adiff < bf16_loss_atol or rdiff < bf16_loss_rtol,
                f"BF16 sonic-moe loss deviates too much from baseline "
                f"at acc_step={step_idx} "
                f"(baseline={loss_base}, bf16={loss_bf16}, "
                f"adiff={adiff:.6e}, rdiff={rdiff:.6e})",
            )

        # ── 7. BF16 sonic-moe vs Baseline: per-step output ──────────────
        for step_idx, (out_base, out_bf16) in enumerate(
            zip(outputs_base, outputs_bf16), start=1
        ):
            diff = calc_diff(out_bf16, out_base)
            print(
                f"BF16 vs Baseline output [acc step {step_idx}/"
                f"{self.acc_steps}]: diff = {diff:.6e}"
            )
            self.assertLess(
                diff,
                1e-4,
                f"BF16 output diff too large at acc_step={step_idx}",
            )

        # ── 8. BF16 sonic-moe vs Baseline: accumulated grad ──────────────
        common_bf16_grads = set(grads_base) & set(grads_bf16)
        self.assertTrue(
            common_bf16_grads,
            "No common BF16 accumulated grad tensors found",
        )
        for name in sorted(common_bf16_grads):
            g0 = grads_base[name]
            g1 = self._aligned_grad_for_compare(
                name, grads_bf16[name], transpose_grouped_gemm=True
            )
            diff = calc_diff(g0, g1)
            print(
                f"BF16 vs Baseline accumulated grad diff = "
                f"{diff:.6e} for {name}"
            )
            self.assertLess(
                diff,
                1e-4,
                f"BF16 accumulated grad diff too large for {name}: "
                f"diff={diff:.6e}",
            )

        # ── 9. FP8 sonic-moe vs BF16: per-step loss ─────────────────────
        fp8_loss_rtol = 1e-2
        fp8_loss_atol = 1e-4
        for step_idx, (loss_bf16, loss_fp8) in enumerate(
            zip(losses_bf16, losses_fp8), start=1
        ):
            adiff = abs(loss_fp8 - loss_bf16)
            rdiff = adiff / max(abs(loss_bf16), 1e-12)
            print(
                f"FP8 vs BF16 [acc step {step_idx}/"
                f"{self.acc_steps}]: loss relative diff = {rdiff:.6e}"
                f", absolute diff = {adiff:.6e}"
            )
            self.assertTrue(
                adiff < fp8_loss_atol or rdiff < fp8_loss_rtol,
                f"FP8 sonic-moe loss deviates too much from BF16 "
                f"at acc_step={step_idx} "
                f"(bf16={loss_bf16}, fp8={loss_fp8}, "
                f"adiff={adiff:.6e}, rdiff={rdiff:.6e})",
            )

        # ── 10. FP8 sonic-moe vs BF16: per-step output ──────────────────
        fp8_tol = 5e-3
        for step_idx, (out_bf16, out_fp8) in enumerate(
            zip(outputs_bf16, outputs_fp8), start=1
        ):
            diff = calc_diff(out_fp8, out_bf16)
            print(
                f"FP8 vs BF16 output [acc step {step_idx}/"
                f"{self.acc_steps}]: diff = {diff:.6e}"
            )
            self.assertLess(
                diff,
                fp8_tol,
                f"FP8 output diff too large at acc_step={step_idx}",
            )

        # ── 11. FP8 sonic-moe vs BF16: accumulated grad ─────────────────
        common_fp8_grads = set(grads_bf16) & set(grads_fp8)
        self.assertTrue(
            common_fp8_grads,
            "No common FP8 accumulated grad tensors found",
        )
        fp8_grad_tol = 5e-3
        for name in sorted(common_fp8_grads):
            g1 = grads_bf16[name]
            g2 = grads_fp8[name]
            diff = calc_diff(g1, g2)
            print(f"FP8 vs BF16 accumulated grad diff = {diff:.6e} for {name}")
            self.assertLess(
                diff,
                fp8_grad_tol,
                f"FP8 accumulated grad diff too large for {name}: "
                f"diff={diff:.6e}, tol={fp8_grad_tol}",
            )

        # ── 12. Per-step intermediate-tensor diff (fwd + bwd) ───────────
        # Only runs when MOE_DEBUG_DUMP=1 was picked up by MoELayer.
        if dumps_base and dumps_bf16 and dumps_fp8:
            self._print_intermediate_diffs(
                "BF16 vs Baseline",
                dumps_base,
                dumps_bf16,
                rhs_is_sonic=True,
            )
            self._print_intermediate_diffs(
                "FP8 vs BF16",
                dumps_bf16,
                dumps_fp8,
                rhs_is_sonic=True,
                lhs_is_sonic=True,
            )

        print("All MoE layer precision comparison checks passed!")


if __name__ == "__main__":
    unittest.main()
