# Solar Open 100B NXD Inference — 실험 결과 보고서

## 개요

`upstage/Solar-Open-100B` 모델을 NeuronX Distributed (NXD) Inference로 실행하기 위한 시도 및 결과를 기록합니다.

- **모델**: `upstage/Solar-Open-100B`
- **아키텍처**: SolarOpenForCausalLM (MoE)
- **인스턴스**: trn2.3xlarge (4 NeuronCore, 96GB HBM total = 24GB/core)
- **실험 날짜**: 2026-02-19

---

## 모델 아키텍처

| 항목 | 값 |
|------|-----|
| `model_type` | `solar_open` |
| `hidden_size` | 4096 |
| `num_hidden_layers` | 48 |
| `num_attention_heads` | 64 |
| `head_dim` | 128 |
| `num_key_value_heads` | 8 |
| `vocab_size` | 196608 |
| `intermediate_size` | 10240 |
| `moe_intermediate_size` | 1280 |
| `n_routed_experts` | 128 |
| `n_shared_experts` | 1 |
| `num_experts_per_tok` | 8 |
| `first_k_dense_replace` | 0 (all layers are MoE) |
| `rope_scaling` | YaRN (factor=2.0, original_max_position_embeddings=65536) |
| `max_position_embeddings` | 131072 |

---

## 구현 내역

### 모델 코드 (`src/neuronx_distributed_inference/models/solar_open/`)

- `__init__.py` — 모듈 초기화
- `modeling_solar_open.py` — 전체 구현
  - `SolarOpenInferenceConfig`: config 로딩 + 누락된 필드(hidden_act, n_group, topk_group) 기본값 처리
  - `NeuronSolarOpenForCausalLM`: `NeuronBaseForCausalLM` 서브클래스
  - `NeuronSolarOpenModel`: 48 MoE 레이어 스택
  - `NeuronSolarOpenDecoderLayer`: attention + MoE MLP
  - `NeuronSolarOpenAttention`: GQA (64 heads → 8 KV heads), YaRN RoPE
  - `initialize_solar_open_moe_module()`: GLM-4.5 MoE와 동일한 구조 (NeuronSolarOpenRouter + ExpertMLPsV2 + SharedExperts)
  - `SolarOpenYarnRotaryEmbedding`: DeepseekV3YarnRotaryEmbedding을 position_ids 인터페이스로 래핑
  - `load_solar_open_config()`: 42개 safetensors 샤드에서 multi-shard weight 변환 (per-expert → NXD 포맷)

### Weight 변환 상세

HF 체크포인트 포맷 (per-expert):
```
mlp.experts.{e}.gate_proj.weight  [moe_intermediate_size, hidden_size]
mlp.experts.{e}.up_proj.weight    [moe_intermediate_size, hidden_size]
mlp.experts.{e}.down_proj.weight  [hidden_size, moe_intermediate_size]
```

NXD 포맷 (fused):
```
mlp.experts.gate_up_proj  [n_experts, hidden_size, 2 * moe_intermediate_size]
mlp.experts.down_proj     [n_experts, moe_intermediate_size, hidden_size]
```

---

## 실험 과정 및 결과

### Phase 1: Tiny Random 모델 테스트 ✅ 성공

- **모델**: 2-layer 랜덤 초기화 Solar Open (128 experts, hidden_size=4096)
- **설정**: `tp_degree=4, moe_tp_degree=4, moe_ep_degree=1`
- **결과**: `test_solar_open_accuracy.py` 10/10 토큰 매칭 통과
- **경로**: `solar_open_tiny_random/` (checkpoint), `solar_open_tiny_random_traced/` (컴파일)

### Phase 2: 실제 100B 모델 테스트 ❌ HBM OOM

#### 시도 1: moe_ep_degree=2 + moe_tp_degree=2

**에러**: EP (Expert Parallelism) + token generation 조합에서 라이브러리 제한
```
NotImplementedError: Selective Loading with Expert parallelism is not supported in token generation.
```
**원인**: `neuronx_distributed.modules.moe.expert_mlps_v2.ExpertMLPsV2.forward()`에서 EP 활성화 시 token generation (seq_len=1)에 selective loading을 시도하지만 EP + selective loading 조합이 미구현 상태.

**해결**: `moe_ep_degree=1, moe_tp_degree=4`로 변경 (EP 제거, TP만 사용)

#### 시도 2: moe_ep_degree=1 + moe_tp_degree=4

**에러**: HBM 메모리 부족 (컴파일 단계)
```
[NCC_EVRF009] Size of total input and output tensors exceeds HBM limit of Trainium2.
Needed 51,370,533,388 bytes (47 GB) vs. available 25,769,803,776 bytes (24 GB).
```

**원인 분석**:

| 항목 | 계산 |
|------|------|
| Expert gate_up weights (48 layers) | 48 × 128 experts × 4096 × 2×1280 × 2 bytes ≈ **102 GB** |
| Expert down weights (48 layers) | 48 × 128 experts × 1280 × 4096 × 2 bytes ≈ **51 GB** |
| Shared expert weights (48 layers) | 48 × 1 × 4096 × 2×10240 × 2 bytes ≈ **8 GB** |
| Attention QKV (48 layers) | 48 × (4096×(64×128 + 2×8×128)) × 2 bytes ≈ **7 GB** |
| **Total** | **~168 GB** |
| tp_degree=4 후 per-core | **~42 GB** |

trn2.3xlarge의 per-core HBM (24 GB)을 2배 초과합니다.

---

## 대형 인스턴스에서의 실행 가이드

### 권장 인스턴스

| 인스턴스 | NeuronCore | HBM | 권장 설정 |
|----------|-----------|-----|----------|
| trn2.3xlarge | 4 | 96 GB | ❌ 불가 (24 GB/core) |
| trn2.48xlarge | 64 | 1.5 TB | ✅ 권장 |
| trn1.32xlarge | 32 | 512 GB | ✅ 가능 |

### trn2.48xlarge 권장 설정

```python
MoENeuronConfig(
    tp_degree=32,          # 32-way tensor parallel
    moe_tp_degree=16,      # MoE expert TP
    moe_ep_degree=2,       # Expert parallelism 가능 (blockwise context encoding 필요)
    batch_size=1,
    ctx_batch_size=1,
    tkg_batch_size=1,
    seq_len=512,
    max_context_length=500,  # 500 * 8 = 4000 > 512 → forward_blockwise 분기
    torch_dtype=torch.bfloat16,
    on_device_sampling_config=OnDeviceSamplingConfig(do_sample=False, top_k=1),
    enable_bucketing=False,
    flash_decoding_enabled=False,
    fused_qkv=True,
    sequence_parallel_enabled=False,
)
```

> **주의**: `moe_ep_degree > 1` 사용 시 `max_context_length * num_experts_per_tok > 512` (default block_size)를 만족해야 context encoding이 EP-지원 `forward_blockwise`로 분기됩니다.

### 컴파일 및 실행

```bash
# trn2.48xlarge에서
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
cd /home/gmkim/neuronx-distributed-inference

# 컴파일 (몇 시간 소요)
python examples/generation_solar_open_100b_demo.py \
    --model-path /path/to/Solar-Open-100B \
    --traced-model-path /path/to/solar_open_100b_traced

# 정확도 테스트
python test_solar_open_100b_accuracy.py \
    --model-path /path/to/Solar-Open-100B \
    --traced-model-path /path/to/solar_open_100b_traced \
    --compile
```

---

## 발견된 라이브러리 제한사항

### 1. EP + Token Generation 미지원

`neuronx_distributed` 라이브러리에서 Expert Parallelism (EP) + token generation (seq_len=1) 조합은 `NotImplementedError`를 발생시킵니다.

**위치**: `ExpertMLPsV2.forward()` line 1458
```python
if self.moe_expert_model_parallel_group.size() > 1:
    raise NotImplementedError(
        "Selective Loading with Expert parallelism is not supported in token generation."
    )
```

**우회 방법**: `moe_ep_degree=1`로 EP를 비활성화하거나, batch_size를 16 이상으로 늘려 `perc_experts_loaded >= 1.0`이 되어 selective loading 분기를 우회.

### 2. Context Encoding에서 EP + forward_all_experts 문제

`max_context_length * top_k <= block_size (512)` 조건에서 context encoding이 `forward_all_experts`를 호출하는데, 이 함수는 EP를 인식하지 못해 global expert 수(128)로 루프를 돌지만 local expert weights(64)만 있어 IndexError 발생.

**우회 방법**: `max_context_length * num_experts_per_tok > 512`를 만족하도록 설정. 또한 scatter 연산에서 `max_context_length % tp_degree == 0` 조건도 만족해야 함.

---

## 파일 목록

| 파일 | 설명 |
|------|------|
| `src/neuronx_distributed_inference/models/solar_open/modeling_solar_open.py` | 전체 모델 구현 |
| `examples/generation_solar_open_100b_demo.py` | 100B 생성 데모 (trn2.3xlarge에서 HBM OOM) |
| `test_solar_open_100b_accuracy.py` | CPU vs Neuron 정확도 테스트 |
| `/home/gmkim/Solar-Open-100B/` | 실제 모델 체크포인트 (42 safetensors 샤드, ~100GB) |

---

## 다음 단계

1. **대형 인스턴스 확보**: trn2.48xlarge 또는 trn1.32xlarge
2. **설정 조정**: 위 권장 설정으로 `examples/generation_solar_open_100b_demo.py` 업데이트
3. **컴파일 및 정확도 검증**: `test_solar_open_100b_accuracy.py` 실행으로 CPU vs Neuron 출력 비교

---

*작성일: 2026-02-19 | 인스턴스: trn2.3xlarge | 모델: upstage/Solar-Open-100B*
