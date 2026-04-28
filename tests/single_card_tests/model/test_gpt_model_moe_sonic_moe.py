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


import functools
import random
import subprocess
import sys
import unittest

import numpy as np
import paddle

paddle.compat.enable_torch_proxy(
    scope={"sonicmoe", "paddlefleet.ops.sonicmoe", "quack", "triton"},
    silent=True,
)
from paddle.distributed import fleet

# from paddlefleet.tensor_parallel.random import model_parallel_cuda_manual_seed
from paddle.distributed.fleet.meta_parallel import NoPipelineParallel
from paddle.distributed.fleet.utils import mix_precision_utils

import paddlefleet

# from tests.unit_tests.test_utilities import Utils
import paddlefleet.parallel_state as ps
from paddlefleet.gpt_builders import gpt_builder
from paddlefleet.models.gpt import GPTConfig
from paddlefleet.ops.utils import get_cuda_version

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


# class TestGPTModel(unittest.TestCase):
#     def setUp(self):
#         seed = 46
#         random.seed(seed)
#         np.random.seed(seed)
#         paddle.seed(seed)
#         strategy = fleet.DistributedStrategy()
#         strategy.hybrid_configs = {
#             "dp_degree": 1,
#             "mp_degree": 1,
#             "pp_degree": 1,
#             "sharding_degree": 1,
#             "sep_degree": 1,
#             "cp_degree": 1,
#             "ep_degree": 1,
#             "moe_sharding_degree": 1,
#             "order": [
#                 "sharding",
#                 "moe_sharding",
#                 "pp",
#                 "sep",
#                 "cp",
#                 "dp",
#                 "ep",
#                 "mp",
#             ],
#         }
#         self.strategy = strategy
#         fleet.init(is_collective=True, strategy=strategy)
#         hcg = fleet.get_hybrid_communicate_group()
#         ps.initialize_model_parallel(hcg)

#         config = GPTConfig(
#             num_hidden_layers=2,
#             hidden_size=512,
#             vocab_size=100,
#             max_sequence_length=64,
#             num_attention_heads=4,
#             intermediate_size=1024,
#             normalization="RMSNorm",
#             hidden_dropout_prob=0.0,
#             attention_dropout=0.0,
#             n_routed_experts=8,
#             use_bias=False,
#             rotary_percent=1.0,
#             rotary_base=10000,
#             rope_scaling=1.0,
#             moe_intermediate_size=1024,
#             moe_token_dispatcher_type="alltoall",
#             n_shared_experts=1,
#             init_method=functools.partial(
#                 paddle.nn.init.xavier_uniform_, gain=1.0
#             ),
#             output_layer_init_method=functools.partial(
#                 paddle.nn.init.xavier_uniform_, gain=1.0
#             ),
#             use_qk_norm=True,
#             moe_grouped_gemm=True,
#             using_sonic_moe=True,
#             fp8="e4m3",
#         )
#         self.gpt_model = gpt_builder(config, num_stages=1)
#         self.config = config

#     def test_forward(self) -> None:
#         sequence_length = self.config.max_sequence_length
#         micro_batch_size = 2

#         for name, param in self.gpt_model.named_parameters():
#             # 计算 L2 范数
#             param_norm = param.detach().norm().item()
#             param_abssum = param.detach().abs().sum().item()
#             print(f"{name}: {param_norm:.6f}, {param_abssum:.6f}")

#         data = list(range(sequence_length))
#         input_ids = paddle.to_tensor(data, dtype=paddle.int64).repeat(
#             (micro_batch_size, 1)
#         )
#         position_ids = paddle.to_tensor(data, dtype=paddle.int64).repeat(
#             (micro_batch_size, 1)
#         )
#         attention_mask = paddle.ones(
#             (micro_batch_size, 1, sequence_length, sequence_length), dtype=bool
#         )
#         labels = paddle.to_tensor(
#             list(range(1, sequence_length + 1)), dtype=paddle.int64
#         ).repeat((micro_batch_size, 1))

#         data = (
#             {
#                 "input_ids": [input_ids],
#                 "position_ids": [position_ids],
#                 "attention_mask": [attention_mask],
#             },
#             [labels],
#         )

#         gpt_pipe_model = NoPipelineParallel(self.gpt_model, self.strategy)
#         gpt_pipe_model = paddle.amp.decorate(
#             models=gpt_pipe_model, level="O2", dtype="bfloat16"
#         )

#         loss = gpt_pipe_model.forward_backward_pipeline(data)

#         for name, param in self.gpt_model.named_parameters():
#             # 计算 L2 范数
#             if param.grad is None:
#                 print(f"{name}: 0.000000, 0.000000")
#                 continue
#             grad_norm = param.grad.detach().norm().item()
#             grad_abssum = param.grad.detach().abs().sum().item()
#             print(f"{name}: {grad_norm:.6f}, {grad_abssum:.6f}")
#             if name == "0.embedding.embed_tokens.weight":
#                 embed_tokens_grad_norm = grad_norm

#         print("loss", loss.item())

#         print("embed_tokens_grad_norm", embed_tokens_grad_norm)

#         clear_all_fp8_weight_caches()
#         repo_name = os.environ.get("repo_flag")
#         if judge_machine_type() == "H":
#             if version == 13:
#                 assert loss.item() == 5.239149570465088, (
#                     f"loss not equal ({loss.item()} != 5.239149570465088), please check your modify"
#                 )
#                 assert embed_tokens_grad_norm == 2.796875, (
#                     f"grad norm of embed_tokens not equal ({embed_tokens_grad_norm} != 2.796875), please check your modify"
#                 )
#             else:  # 12.X
#                 if cuda_minor == 6:
#                     assert loss.item() == 5.239708423614502, (
#                         f"loss not equal ({loss.item()} != 5.239708423614502), please check your modify"
#                     )
#                     assert embed_tokens_grad_norm == 2.796875, (
#                         f"grad norm of embed_tokens not equal ({embed_tokens_grad_norm} != 2.796875), please check your modify"
#                     )
#                 else:  # 12.9
#                     assert loss.item() == 5.239149570465088, (
#                         f"loss not equal ({loss.item()} != 5.239149570465088), please check your modify"
#                     )
#                     assert embed_tokens_grad_norm == 2.796875, (
#                         f"grad norm of embed_tokens not equal ({embed_tokens_grad_norm} != 2.796875), please check your modify"
#                     )
#         elif judge_machine_type() == "V":
#             pass  # TODO: add V machine test


@unittest.skipUnless(
    paddlefleet.ops.is_sonic_moe_available(),
    "Sonic-MoE not available (requires Python>=3.12, CUDA>=12.9, SM>=90)",
)
class TestSonicMoEPrecision(unittest.TestCase):
    """Precision comparison: baseline grouped_gemm vs BF16 sonic-moe vs FP8 sonic-moe."""

    def setUp(self):
        strategy = fleet.DistributedStrategy()
        strategy.hybrid_configs = {
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
        self.strategy = strategy
        fleet.init(is_collective=True, strategy=strategy)
        hcg = fleet.get_hybrid_communicate_group()
        ps.initialize_model_parallel(hcg)

    def _base_config_kwargs(self):
        """Common config kwargs shared by all variants."""
        return {
            "num_hidden_layers": 2,
            "hidden_size": 512,
            "vocab_size": 100,
            "max_sequence_length": 64,
            "num_attention_heads": 4,
            "intermediate_size": 1024,
            "normalization": "RMSNorm",
            "hidden_dropout_prob": 0.0,
            "attention_dropout": 0.0,
            "n_routed_experts": 8,
            "use_bias": False,
            "rotary_percent": 1.0,
            "rotary_base": 10000,
            "rope_scaling": 1.0,
            "moe_intermediate_size": 1024,
            "moe_token_dispatcher_type": "alltoall",
            "n_shared_experts": 1,
            "init_method": functools.partial(
                paddle.nn.init.xavier_uniform_, gain=1.0
            ),
            "output_layer_init_method": functools.partial(
                paddle.nn.init.xavier_uniform_, gain=1.0
            ),
            "use_qk_norm": True,
            # All variants use gated_linear_unit=True so that the baseline
            # GroupedMLPExpert also computes SwiGLU, matching sonic-moe's
            # internal activation.  This ensures a fair comparison.
            "gated_linear_unit": True,
        }

    def _build_model(self, seed=46, **extra):
        """Build a GPT MoE model with deterministic initialization."""
        random.seed(seed)
        np.random.seed(seed)
        paddle.seed(seed)
        kw = self._base_config_kwargs()
        kw.update(extra)
        config = GPTConfig(**kw)
        model = gpt_builder(config, num_stages=1)
        return config, model

    def _make_data(self, config, micro_batch_size=2):
        """Create deterministic input data."""
        seq = config.max_sequence_length
        ids = list(range(seq))
        input_ids = paddle.to_tensor(ids, dtype=paddle.int64).repeat(
            (micro_batch_size, 1)
        )
        position_ids = paddle.to_tensor(ids, dtype=paddle.int64).repeat(
            (micro_batch_size, 1)
        )
        attention_mask = paddle.ones(
            (micro_batch_size, 1, seq, seq), dtype=bool
        )
        labels = paddle.to_tensor(
            list(range(1, seq + 1)), dtype=paddle.int64
        ).repeat((micro_batch_size, 1))
        return (
            {
                "input_ids": [input_ids],
                "position_ids": [position_ids],
                "attention_mask": [attention_mask],
            },
            [labels],
        )

    def _forward_backward(self, model, data):
        """Run forward + backward and return (loss_value, {param_name: grad_tensor})."""
        pipe = NoPipelineParallel(model, self.strategy)
        pipe = paddle.amp.decorate(
            models=pipe,
            level="O2",
            dtype="bfloat16",
            master_grad=True,
            master_weight=True,
        )
        mix_precision_utils.MixPrecisionLayer(pipe, dtype="bfloat16")

        with paddle.amp.auto_cast(level="O2", dtype="bfloat16"):
            loss = pipe.forward_backward_pipeline(data)

        grads = {}
        for name, p in pipe.named_parameters():
            grad = getattr(p, "main_grad", None)
            if grad is not None:
                grads[name] = grad.detach().clone()
        return loss.item(), grads

    @staticmethod
    def _split_to_sonic_interleaved(weight):
        gate, up = paddle.chunk(weight, 2, axis=-1)
        gate = gate.transpose([0, 2, 1])
        up = up.transpose([0, 2, 1])
        return paddle.stack([gate, up], axis=2).reshape(
            weight.shape[0], -1, weight.shape[1]
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
    def _copy_weights_to_sonic(cls, src_model, dst_model):
        """Copy weights from a non-sonic-moe model to a sonic-moe model.

        GroupedMLPExpert weight layouts differ by using_sonic_moe:
          non-sonic: w1=[E, hidden, fc1_out], w2=[E, fc2_in, hidden]
          sonic:     w1=[E, fc1_out, hidden], w2=[E, hidden, fc2_in]
        Sonic-MoE stores w1 rows interleaved as gate/up pairs, so w1 is
        transposed and interleaved. w2 is only transposed on axes [1, 2].
        All other weights (embedding, attention, norm, shared expert) share
        the same shape and are copied directly.
        """
        src_params = dict(src_model.named_parameters())
        for name, dst_p in dst_model.named_parameters():
            src_p = src_params[name]
            if "grouped_gemm_experts.weight1" in name:
                dst_p.set_value(cls._split_to_sonic_interleaved(src_p))
            elif "grouped_gemm_experts.weight2" in name:
                dst_p.set_value(src_p.transpose([0, 2, 1]))
            else:
                dst_p.set_value(src_p.clone())

    def test_precision_comparison(self) -> None:
        # ── 1. Baseline: grouped_gemm BF16 (no sonic-moe) ──────────────
        cfg_base, model_base = self._build_model(
            moe_grouped_gemm=True,
            using_sonic_moe=False,
        )

        # ── 2. BF16 sonic-moe ──────────────────────────────────────────
        cfg_bf16, model_bf16 = self._build_model(
            moe_grouped_gemm=True,
            using_sonic_moe=True,
        )
        self._copy_weights_to_sonic(model_base, model_bf16)

        # ── 3. FP8 sonic-moe ──────────────────────────────────────────
        cfg_fp8, model_fp8 = self._build_model(
            moe_grouped_gemm=True,
            using_sonic_moe=True,
            fp8="e4m3",
        )
        self._copy_weights_to_sonic(model_base, model_fp8)

        # ── Run forward + backward for each variant ────────────────────
        loss_base, grads_base = self._forward_backward(
            model_base, self._make_data(cfg_base)
        )
        print(f"[Baseline]       loss = {loss_base}")

        loss_bf16, grads_bf16 = self._forward_backward(
            model_bf16, self._make_data(cfg_bf16)
        )
        print(f"[BF16 sonic-moe] loss = {loss_bf16}")

        loss_fp8, grads_fp8 = self._forward_backward(
            model_fp8, self._make_data(cfg_fp8)
        )
        clear_all_fp8_weight_caches()
        print(f"[FP8  sonic-moe] loss = {loss_fp8}")

        # ── Check BF16 sonic-moe vs Baseline ───────────────────────────
        rdiff_bf16 = abs(loss_bf16 - loss_base) / abs(loss_base)
        print(f"BF16 vs Baseline: loss relative diff = {rdiff_bf16:.6e}")
        self.assertLess(
            rdiff_bf16,
            1e-2,
            f"BF16 sonic-moe loss deviates too much from baseline "
            f"(baseline={loss_base}, bf16_sonic={loss_bf16})",
        )

        common_bf16_grads = set(grads_base) & set(grads_bf16)
        self.assertTrue(
            common_bf16_grads, "No common BF16 main_grad tensors found"
        )
        for name in sorted(common_bf16_grads):
            g0 = grads_base[name]
            g1 = self._aligned_grad_for_compare(
                name, grads_bf16[name], transpose_grouped_gemm=True
            )
            diff = calc_diff(g0, g1)
            print(f"BF16 vs Baseline: grad diff = {diff:.6e} for {name}")
            self.assertLess(
                diff,
                5e-5,
                f"BF16 grad tensor diff too large for {name}: diff={diff:.6e}",
            )

        # ── Check FP8 sonic-moe vs BF16 sonic-moe ─────────────────────
        rdiff_fp8 = abs(loss_fp8 - loss_bf16) / abs(loss_bf16)
        print(f"FP8 vs BF16: loss relative diff = {rdiff_fp8:.6e}")
        self.assertLess(
            rdiff_fp8,
            0.1,
            f"FP8 sonic-moe loss deviates too much from BF16 "
            f"(bf16={loss_bf16}, fp8={loss_fp8})",
        )

        common_fp8_grads = set(grads_bf16) & set(grads_fp8)
        self.assertTrue(
            common_fp8_grads, "No common FP8 main_grad tensors found"
        )
        for name in sorted(common_fp8_grads):
            g1 = grads_bf16[name]
            g2 = grads_fp8[name]
            diff = calc_diff(g1, g2)
            # grad_tol = 0.5 if "grouped_gemm_experts.weight" in name else 0.2
            grad_tol = 1e-3
            print(f"FP8 vs BF16: grad diff = {diff:.6e} for {name}")
            self.assertLess(
                diff,
                grad_tol,
                f"FP8 grad tensor diff too large for {name}: "
                f"diff={diff:.6e}, tol={grad_tol}",
            )

        print("All precision comparison checks passed!")


if __name__ == "__main__":
    unittest.main()
