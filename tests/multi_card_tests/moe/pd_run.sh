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

unset PADDLE_ELASTIC_JOB_ID
unset PADDLE_TRAINER_ENDPOINTS
unset DISTRIBUTED_TRAINER_ENDPOINTS
unset FLAGS_START_PORT
unset PADDLE_ELASTIC_TIMEOUT

nnodes=$PADDLE_TRAINERS_NUM
rank=$PADDLE_TRAINER_ID

for name in `env | grep -E 'PADDLE|ENDPOINT' | awk -F'=' '{print $1}'`; do
  unset ${name}
done

source /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/PaddleFleet/.venv/bin/activate

export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
export USE_QUACK_GEMM=1
export QUACK_CACHE_DIR=/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/output/quack_cache
export FLAGS_cudnn_deterministic=1
export FLAGS_embedding_deterministic=1

rm -rf /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/output/ut_output_$rank
python -m paddle.distributed.launch \
  --log_dir /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/output/ut_output_$rank/paddle_distributed_logs \
  --gpus="0,1,2,3,4,5,6,7" \
  --run_mode=collective \
  test_sonic_moe_ep.py
