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
import sys
import unittest

# Enable intermediate-tensor dumps in MoELayer before importing it so
# the module-level env check flips on for the whole test process.
os.environ.setdefault("MOE_DEBUG_DUMP", "1")

import numpy as np
import paddle
import paddle.nn.functional as F

paddle.compat.enable_torch_proxy(
    scope={"sonicmoe", "paddlefleet.ops.sonicmoe", "quack", "triton"},
    silent=True,
)
from paddle.distributed import fleet

import paddlefleet
from paddlefleet.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
)
from paddlefleet.process_groups_config import ProcessGroupCollection
from paddlefleet.tensor_parallel.random import model_parallel_cuda_manual_seed
from paddlefleet.training.global_vars import unset_global_variables
from paddlefleet.training.initialize import initialize_fleet
from paddlefleet.transformer.moe.moe_layer import MoELayer
from paddlefleet.transformer.transformer_config import TransformerConfig

if paddlefleet.ops.is_sonic_moe_available():
    from paddlefleet.ops.sonicmoe.functional import clear_all_fp8_weight_caches

for _key in list(sys.modules):
    if _key.startswith("paddlefleet.ops.sonicmoe"):
        _alias = _key.replace("paddlefleet.ops.sonicmoe", "sonicmoe", 1)
        sys.modules.setdefault(_alias, sys.modules[_key])


@unittest.skipUnless(
    paddlefleet.ops.is_sonic_moe_available(),
    "Sonic-MoE not available (requires Python>=3.12, CUDA>=12.9, SM>=90)",
)
class TestSonicMoEExpertParallelPrecision(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        strategy = fleet.DistributedStrategy()
        strategy.hybrid_configs = {
            "dp_degree": 1,
            "mp_degree": 4,
            "pp_degree": 1,
            "sharding_degree": 2,
            "sep_degree": 1,
            "cp_degree": 1,
            "ep_degree": 4,
            "moe_sharding_degree": 2,
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
        initialize_fleet(strategy=strategy)
        cls.pg_collection = ProcessGroupCollection.use_mpu_process_groups()

    @classmethod
    def tearDownClass(cls):
        unset_global_variables()

    def setUp(self):
        self.seed = 123
        self.hidden_size = 256
        self.n_routed_experts = 64

        random.seed(self.seed)
        np.random.seed(self.seed)
        paddle.seed(self.seed)
        paddle.manual_seed(self.seed)
        model_parallel_cuda_manual_seed(self.seed)
        self.pg_collection = self.__class__.pg_collection

    @staticmethod
    def calc_diff(x: paddle.Tensor, y: paddle.Tensor):
        x, y = x.double(), y.double()
        denominator = (x * x + y * y).sum()
        if denominator.item() == 0:
            return 0.0
        sim = 2 * (x * y).sum() / denominator
        return (1 - sim).item()

    def _build_transformer_config(
        self,
        using_sonic_moe=False,
        fp8=None,
        moe_deep_gemm=False,
        fp8_wgrad=True,
    ):
        return TransformerConfig(
            hidden_size=self.hidden_size,
            num_attention_heads=4,
            n_routed_experts=self.n_routed_experts,
            use_cpu_initialization=False,
            num_experts_per_tok=2,
            tensor_model_parallel_size=1,
            expert_model_parallel_size=4,
            sequence_parallel=False,
            bf16=True,
            params_dtype=paddle.bfloat16,
            moe_intermediate_size=128,
            gated_linear_unit=True,
            n_shared_experts=0,
            hidden_act=F.silu,
            moe_grouped_gemm=True,
            moe_deep_gemm=moe_deep_gemm,
            bias_activation_fusion=True,
            moe_token_dispatcher_type="deepep",
            moe_use_fusion_node=True,
            using_sonic_moe=using_sonic_moe,
            fp8=fp8,
            fp8_wgrad=fp8_wgrad,
        )

    def _build_moe_layer(
        self,
        using_sonic_moe=False,
        fp8=None,
        moe_deep_gemm=False,
        fp8_wgrad=True,
    ):
        transformer_config = self._build_transformer_config(
            using_sonic_moe=using_sonic_moe,
            fp8=fp8,
            moe_deep_gemm=moe_deep_gemm,
            fp8_wgrad=fp8_wgrad,
        )
        transformer_layer_spec = get_gpt_layer_local_spec(
            transformer_config,
            num_experts=self.n_routed_experts,
        )
        return MoELayer(
            transformer_config,
            transformer_layer_spec.sublayers_spec.mlp.extra_kwargs["sublayers"],
            self.pg_collection,
        )

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

    @staticmethod
    def _swap_halves(grad, axis):
        if grad.shape[axis] % 2 != 0:
            return grad
        first, second = paddle.chunk(grad, 2, axis=axis)
        return paddle.concat([second, first], axis=axis)

    def _aligned_deep_gemm_bf16_wgrad(self, name, grad, ref_grad=None):
        if "grouped_gemm_experts.weight" not in name:
            return grad

        candidates = [("raw", grad)]
        if len(grad.shape) == 3:
            candidates.append(("transpose", grad.transpose([0, 2, 1])))

        if "grouped_gemm_experts.weight1" in name:
            expanded = []
            for tag, candidate in candidates:
                expanded.extend(
                    [
                        (tag, candidate),
                        (
                            f"{tag}_swap_last",
                            self._swap_halves(candidate, axis=-1),
                        ),
                        (
                            f"{tag}_swap_mid",
                            self._swap_halves(candidate, axis=1),
                        ),
                    ]
                )
            candidates = expanded

        if ref_grad is None:
            for tag, candidate in candidates:
                if tag == "transpose":
                    return candidate
            return candidates[0][1]

        best_tag, best_grad = candidates[0]
        best_diff = None
        for tag, candidate in candidates:
            if list(candidate.shape) != list(ref_grad.shape):
                continue
            diff = self.calc_diff(candidate, ref_grad)
            if best_diff is None or diff < best_diff:
                best_tag, best_grad, best_diff = tag, candidate, diff
        print(
            f"[DeepGEMM align] {name}: selected {best_tag}"
            + (f" diff = {best_diff:.6e}" if best_diff is not None else "")
        )
        return best_grad

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
        # moe_layer = paddle.amp.decorate(
        #     models=moe_layer,
        #     level="O2",
        #     dtype="bfloat16",
        #     master_grad=True,
        #     master_weight=True,
        # )
        # mix_precision_utils.MixPrecisionLayer(moe_layer, dtype="bfloat16")
        hidden_states = input_data.detach().clone()
        hidden_states.stop_gradient = False
        with paddle.amp.auto_cast(level="O2", dtype="bfloat16"):
            output = moe_layer(hidden_states)[0]
            loss = output.sum()
            # loss = paddle.mean(paddle.square(output.cast("float32")))
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

        Args:
            moe_layer: The MoE layer to test.
            input_data_list: List of input tensors, one per accumulation step.
            inner_moe_layer: Optional inner MoELayer whose
                ``_debug_dump_enabled`` / ``_debug_dump`` attributes are read
                to collect per-step intermediate-tensor snapshots.

        Returns:
            losses: List of loss values for each step.
            outputs: List of output tensors for each step.
            grads: Dict of accumulated gradient tensors after all steps.
            dumps: List of per-step {"fwd": {...}, "bwd": {...}} snapshots
                (empty when dumping is disabled).
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
        """Diff that tolerates shape/dtype mismatches.

        Returns None when tensors are truly incompatible.
        """
        try:
            if a.shape != b.shape:
                return None
            return TestSonicMoEExpertParallelPrecision.calc_diff(a, b)
        except Exception:
            return None

    # Execution order of forward intermediates within MoELayer.  Keys
    # that don't exist on a given path are skipped during diff printing.
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

    def _print_intermediate_diffs(self, label, dumps_lhs, dumps_rhs):
        """Diff forward + backward intermediate tensors step by step.

        Forward follows ``_FWD_EXEC_ORDER``; backward uses the reverse
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

    def _assert_loss_close(self, lhs, rhs, tol, title):
        loss_rdiff = abs(lhs - rhs) / max(abs(rhs), 1e-12)
        print(f"{title}: loss relative diff = {loss_rdiff:.6e}")
        self.assertLess(
            loss_rdiff,
            tol,
            f"{title} loss deviates too much: lhs={lhs}, rhs={rhs}",
        )

    def _assert_tensor_diff_less(self, lhs, rhs, tol, title):
        diff = self.calc_diff(lhs, rhs)
        print(f"{title}: diff = {diff:.6e}")
        self.assertLess(diff, tol, f"{title} diff too large: diff={diff:.6e}")

    def _assert_grad_diff_less(
        self,
        lhs_grads,
        rhs_grads,
        tol,
        title,
        transpose_grouped_gemm=False,
    ):
        lhs_names = set(lhs_grads)
        rhs_names = set(rhs_grads)
        self.assertEqual(
            lhs_names,
            rhs_names,
            (
                f"Gradient tensors mismatch for {title}: "
                f"lhs_only={sorted(lhs_names - rhs_names)}, "
                f"rhs_only={sorted(rhs_names - lhs_names)}"
            ),
        )
        self.assertTrue(lhs_names, f"No grad tensors found for {title}")
        for name in sorted(lhs_names):
            lhs_grad = self._aligned_grad_for_compare(
                name,
                lhs_grads[name],
                transpose_grouped_gemm=transpose_grouped_gemm,
            )
            grad_tol = tol[name] if isinstance(tol, dict) else tol
            self._assert_tensor_diff_less(
                lhs_grad,
                rhs_grads[name],
                tol=grad_tol,
                title=f"{title} grad {name}",
            )

    # def test_sonic_moe_ep_precision(self):
    #     moe_layer_base = self._build_moe_layer(using_sonic_moe=False)
    #     moe_layer_sonic_bf16 = self._build_moe_layer(using_sonic_moe=True)
    #     moe_layer_sonic_fp8 = self._build_moe_layer(
    #         using_sonic_moe=True,
    #         fp8="e4m3",
    #     )
    #     self._copy_weights(
    #         moe_layer_base,
    #         moe_layer_sonic_bf16,
    #         transpose_grouped_gemm=True,
    #     )
    #     self._copy_weights(
    #         moe_layer_base,
    #         moe_layer_sonic_fp8,
    #         transpose_grouped_gemm=True,
    #     )

    #     input_data = paddle.randn(
    #         [4, 256, self.hidden_size],
    #         dtype=paddle.bfloat16,
    #     )

    #     (
    #         output_base,
    #         loss_base,
    #         input_grad_base,
    #         grads_base,
    #     ) = self._run_forward_backward(moe_layer_base, input_data)
    #     print(f"[Baseline]           loss = {loss_base}")

    #     (
    #         output_sonic_bf16,
    #         loss_sonic_bf16,
    #         input_grad_sonic_bf16,
    #         grads_sonic_bf16,
    #     ) = self._run_forward_backward(moe_layer_sonic_bf16, input_data)
    #     print(f"[Sonic-MoE BF16]    loss = {loss_sonic_bf16}")

    #     (
    #         output_sonic_fp8,
    #         loss_sonic_fp8,
    #         input_grad_sonic_fp8,
    #         grads_sonic_fp8,
    #     ) = self._run_forward_backward(moe_layer_sonic_fp8, input_data)
    #     clear_all_fp8_weight_caches()
    #     print(f"[Sonic-MoE FP8]     loss = {loss_sonic_fp8}")

    #     self._assert_loss_close(
    #         loss_sonic_bf16,
    #         loss_base,
    #         tol=1e-2,
    #         title="Sonic-MoE BF16 vs Baseline",
    #     )
    #     self._assert_tensor_diff_less(
    #         output_sonic_bf16,
    #         output_base,
    #         tol=1e-4,
    #         title="Sonic-MoE BF16 vs Baseline output",
    #     )
    #     self._assert_tensor_diff_less(
    #         input_grad_sonic_bf16,
    #         input_grad_base,
    #         tol=1e-4,
    #         title="Sonic-MoE BF16 vs Baseline input grad",
    #     )
    #     self._assert_grad_diff_less(
    #         grads_sonic_bf16,
    #         grads_base,
    #         tol=1e-4,
    #         title="Sonic-MoE BF16 vs Baseline",
    #         transpose_grouped_gemm=True,
    #     )

    #     fp8_tol = 5e-3
    #     self._assert_loss_close(
    #         loss_sonic_fp8,
    #         loss_sonic_bf16,
    #         tol=0.1,
    #         title="Sonic-MoE FP8 vs BF16",
    #     )
    #     self._assert_tensor_diff_less(
    #         output_sonic_fp8,
    #         output_sonic_bf16,
    #         tol=fp8_tol,
    #         title="Sonic-MoE FP8 vs BF16 output",
    #     )
    #     self._assert_tensor_diff_less(
    #         input_grad_sonic_fp8,
    #         input_grad_sonic_bf16,
    #         tol=fp8_tol,
    #         title="Sonic-MoE FP8 vs BF16 input grad",
    #     )
    #     self._assert_grad_diff_less(
    #         grads_sonic_fp8,
    #         grads_sonic_bf16,
    #         tol=fp8_tol,
    #         title="Sonic-MoE FP8 vs BF16",
    #     )

    def test_sonic_moe_ep_grad_accumulation(self):
        """Test loss and grad precision under gradient accumulation
        with expert parallelism: baseline vs BF16 sonic-moe vs FP8 sonic-moe.
        """
        acc_steps = 1

        # ── 1. Build models ─────────────────────────────────────────────
        moe_layer_base = self._build_moe_layer(using_sonic_moe=False)
        moe_layer_sonic_bf16 = self._build_moe_layer(using_sonic_moe=True)
        moe_layer_sonic_fp8 = self._build_moe_layer(
            using_sonic_moe=True,
            fp8="e4m3",
        )

        # ── 2. Copy weights from baseline to sonic variants ─────────────
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

        # ── 3. Generate input data for each accumulation step ───────────
        input_data_list = []
        for step_idx in range(acc_steps):
            paddle.seed(self.seed + step_idx)
            data = paddle.randn(
                [4, 256, self.hidden_size],
                dtype=paddle.bfloat16,
            )
            input_data_list.append(data)

        # ── 4. Run accumulated forward-backward ─────────────────────────
        losses_base, outputs_base, grads_base, dumps_base = (
            self._run_accumulated_forward_backward(
                moe_layer_base, input_data_list, moe_layer_base
            )
        )
        print(f"[Baseline] accumulated losses = {losses_base}")

        losses_bf16, outputs_bf16, grads_bf16, dumps_bf16 = (
            self._run_accumulated_forward_backward(
                moe_layer_sonic_bf16, input_data_list, moe_layer_sonic_bf16
            )
        )
        print(f"[Sonic-MoE BF16] accumulated losses = {losses_bf16}")

        losses_fp8, outputs_fp8, grads_fp8, dumps_fp8 = (
            self._run_accumulated_forward_backward(
                moe_layer_sonic_fp8, input_data_list, moe_layer_sonic_fp8
            )
        )
        clear_all_fp8_weight_caches()
        print(f"[Sonic-MoE FP8] accumulated losses = {losses_fp8}")

        # ── 5. Verify step counts ──────────────────────────────────────
        self.assertEqual(len(losses_base), acc_steps)
        self.assertEqual(len(losses_bf16), acc_steps)
        self.assertEqual(len(losses_fp8), acc_steps)

        # ── 5b. Per-step intermediate-tensor diff (fwd + bwd) ──────────
        # Printed before any tolerance assertion so diffs are visible
        # even when a later assertion fails.  Only runs when
        # MOE_DEBUG_DUMP=1 was picked up by MoELayer.
        if dumps_base and dumps_bf16 and dumps_fp8:
            self._print_intermediate_diffs(
                "BF16 vs Baseline", dumps_base, dumps_bf16
            )
            self._print_intermediate_diffs("FP8 vs BF16", dumps_bf16, dumps_fp8)

        # ── 6. BF16 sonic-moe vs Baseline: per-step loss ────────────────
        for step_idx, (loss_base, loss_bf16) in enumerate(
            zip(losses_base, losses_bf16), start=1
        ):
            self._assert_loss_close(
                loss_bf16,
                loss_base,
                tol=1e-3,
                title=(
                    f"Sonic-MoE BF16 vs Baseline "
                    f"[acc step {step_idx}/{acc_steps}]"
                ),
            )

        # ── 6b. BF16 sonic-moe vs Baseline: per-step output tensor ─────
        for step_idx, (out_base, out_bf16) in enumerate(
            zip(outputs_base, outputs_bf16), start=1
        ):
            self._assert_tensor_diff_less(
                out_bf16,
                out_base,
                tol=1e-2,
                title=(
                    f"Sonic-MoE BF16 vs Baseline output "
                    f"[acc step {step_idx}/{acc_steps}]"
                ),
            )

        # ── 7. BF16 sonic-moe vs Baseline: accumulated grad ────────────
        self._assert_grad_diff_less(
            grads_bf16,
            grads_base,
            tol=1e-2,
            title="Sonic-MoE BF16 vs Baseline accumulated grad",
            transpose_grouped_gemm=True,
        )

        # ── 8. FP8 sonic-moe vs BF16: per-step loss ────────────────────
        for step_idx, (loss_bf16, loss_fp8) in enumerate(
            zip(losses_bf16, losses_fp8), start=1
        ):
            self._assert_loss_close(
                loss_fp8,
                loss_bf16,
                tol=1e-2,
                title=(
                    f"Sonic-MoE FP8 vs BF16 [acc step {step_idx}/{acc_steps}]"
                ),
            )

        # ── 8b. FP8 sonic-moe vs BF16: per-step output tensor ──────────
        fp8_tol = 5e-3
        for step_idx, (out_bf16, out_fp8) in enumerate(
            zip(outputs_bf16, outputs_fp8), start=1
        ):
            self._assert_tensor_diff_less(
                out_fp8,
                out_bf16,
                tol=fp8_tol,
                title=(
                    f"Sonic-MoE FP8 vs BF16 output "
                    f"[acc step {step_idx}/{acc_steps}]"
                ),
            )

        # ── 9. FP8 sonic-moe vs BF16: accumulated grad ─────────────────
        fp8_grad_tol = 5e-3
        self._assert_grad_diff_less(
            grads_fp8,
            grads_bf16,
            tol=fp8_grad_tol,
            title="Sonic-MoE FP8 vs BF16 accumulated grad",
        )

        print("All gradient accumulation precision checks passed!")


if __name__ == "__main__":
    unittest.main()
