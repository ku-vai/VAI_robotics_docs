---
sidebar_position: 1
slug: /policy-training
title: ACT 정책 학습
---

# lerobot-train으로 ACT 정책 학습하기

이 예제는 **CUDA GPU가 있는 Ubuntu 학습 PC**에서 로봇 연결 없이 ACT 정책을 학습하는 과정을 다룹니다.

## 1. 학습 PC 준비

:::tip[이미 셋업된 연구실 PC라면]
아래 서브모듈 초기화는 건너뛰고 학습에 사용할 conda 환경과 데이터셋 경로를 확인하세요.
:::

`VAI_AIWORKER/lerobot`은 Git 서브모듈입니다. 저장소를 클론한 뒤 `lerobot/`이 비어 있다면 **`VAI_AIWORKER` 루트에서** 아래 명령을 실행합니다.

```bash
git submodule update --init -- lerobot
git submodule status -- lerobot
```

첫 명령은 `VAI_AIWORKER`가 지정한 커밋의 소스를 `lerobot/`에 받습니다. 두 번째 명령의 출력에서 커밋 해시 앞이 공백이면 해당 버전이 준비된 상태이고, `-`이면 아직 초기화되지 않은 상태입니다.

- **학습 환경** — `conda run -n lerobot_act`로 해당 환경의 `lerobot-train`을 실행합니다. 환경 이름은 `conda env list`에서 확인할 수 있습니다. 다른 PC에서는 `-n` 뒤의 환경 이름과 프로젝트·데이터셋 경로를 해당 PC에 맞추세요.
- **학습 데이터** — `/home/vai/aiworker_data/pick_and_place_fine_arm16` 아래에 `meta/`, `data/`, `videos/`가 있어야 합니다. 팔·그리퍼 16관절로 가공된 데이터를 사용하며, 관절 순서와 입력 카메라는 데이터셋의 `meta/info.json`에서 확인할 수 있습니다. [녹화와 데이터셋 검수](../5_vr-teleoperation/4_recording.md#데이터셋-검수)

## 2. ACT 학습 스크립트

기본 예제는 W&B 계정이나 로그인 없이 실행할 수 있습니다. `conda run`의 `--no-capture-output` 옵션으로 학습 중 출력과 오류를 터미널과 로그 파일에서 바로 확인합니다.

:::warning[새 학습에는 새 출력 폴더를 지정하세요]
`--output_dir` 폴더가 이미 있으면 새 학습은 `FileExistsError`로 중단됩니다. 새 실험에는 아직 존재하지 않는 경로를 지정하세요. 중단된 실험을 이어가려면 3절의 재개 스크립트를 사용하세요.
:::

```bash title="train_act_arm16.sh"
#!/bin/bash
set -euo pipefail

cd /home/vai/Desktop/yujin/VAI_AIWORKER/lerobot
mkdir -p outputs/train

conda run -n lerobot_act --no-capture-output lerobot-train \
  --dataset.repo_id=local/pick_and_place_fine_arm16 \
  --dataset.root=/home/vai/aiworker_data/pick_and_place_fine_arm16 \
  --dataset.video_backend=pyav \
  --policy.type=act --policy.device=cuda --policy.push_to_hub=false \
  --policy.optimizer_lr=5e-5 --policy.optimizer_lr_backbone=5e-5 \
  --output_dir=outputs/train/act_pick_and_place_arm16_bs256_50k \
  --job_name=act_pick_and_place_arm16_bs256_50k \
  --batch_size=256 --steps=50000 --save_freq=2000 --log_freq=50 --num_workers=16 \
  --wandb.enable=false \
  2>&1 | tee -a outputs/train/act_pick_and_place_arm16_bs256_50k.log
```

### 학습 인자 설명

| 인자 | 이 예제에서 하는 일 |
| --- | --- |
| `--dataset.repo_id` | 학습할 데이터셋의 식별자입니다. 팀에서 가공한 로컬 데이터셋은 `local/pick_and_place_fine_arm16`처럼 지정합니다. |
| `--dataset.root` | 학습 PC에서 실제 파일을 읽을 경로입니다. `meta/info.json`이 들어 있는 데이터셋 폴더를 지정합니다. |
| `--dataset.video_backend` | 카메라 영상을 읽는 디코더입니다. 이 예제는 `pyav`를 사용합니다. |
| `--policy.type` | 학습할 정책을 선택합니다. `act`는 앞으로의 관절 명령 여러 개를 한 번에 예측합니다. |
| `--policy.device` | 학습에 사용할 장치입니다. `cuda`는 NVIDIA GPU를 사용합니다. |
| `--policy.push_to_hub` | 학습 후 모델을 Hugging Face Hub에 올릴지 정합니다. `false`이면 로컬에만 저장합니다. |
| `--policy.optimizer_lr` | 정책의 기본 학습률입니다. 한 번의 업데이트에서 가중치를 얼마나 바꿀지 조절합니다. 예제 값은 `5e-5`입니다. |
| `--policy.optimizer_lr_backbone` | 카메라 영상에서 특징을 추출하는 백본에 적용할 학습률입니다. 예제에서는 같은 값인 `5e-5`를 씁니다. |
| `--output_dir` | 체크포인트와 학습 결과를 저장할 폴더입니다. |
| `--job_name` | 실행을 구분하는 이름입니다. 예제에서는 출력 폴더 이름과 맞춰 둡니다. |
| `--batch_size` | 한 번의 업데이트에 사용할 샘플 수입니다. 예제는 GPU 한 대에서 샘플 256개를 한 배치로 처리합니다. |
| `--steps` | 학습할 총 업데이트 횟수입니다. `50000`이면 50,000스텝에서 학습을 마칩니다. |
| `--save_freq` | 체크포인트 저장 간격입니다. `2000`이면 2,000스텝마다 저장합니다. |
| `--log_freq` | 학습 지표를 출력하는 간격입니다. `50`이면 50스텝마다 손실값 등을 출력합니다. |
| `--num_workers` | 학습용 배치를 읽고 준비하는 데이터 로더 프로세스 수입니다. 예제 값은 `16`입니다. |
| `--wandb.enable` | W&B에 학습 지표를 기록할지 정합니다. 기본 예제는 `false`이며 지표는 터미널과 로컬 로그에서 확인합니다. |

:::info[다른 로컬 데이터셋으로 바꿀 때]
`--dataset.repo_id`와 `--dataset.root`를 함께 바꾸세요. 예를 들어 블록 쌓기 데이터셋이라면 각각 `local/stack_blocks_arm16`과 `/home/vai/aiworker_data/stack_blocks_arm16`을 지정합니다. 같은 조건으로 비교할 때는 배치 크기, 학습률, 총 스텝을 그대로 사용합니다.
:::

새 실행의 이름을 `act_stack_blocks_arm16_bs256_50k`로 정했다면 `--job_name`에 그 이름을 쓰고, `--output_dir`에는 `outputs/train/act_stack_blocks_arm16_bs256_50k`를 지정하세요. 스크립트 끝의 `tee -a`는 터미널 출력을 파일에도 덧붙이는 셸 명령이므로, 로그 파일명도 새 이름으로 바꿉니다. 이 학습을 재개할 때는 3절의 `RUN_DIR`을 새 `--output_dir`와 같은 경로로 설정합니다.

관절·카메라 구성이 바뀌었다면 추론 시에도 새 데이터셋의 `meta/info.json`을 기준으로 관절 순서와 `EXCLUDED_JOINTS`, `CAMERAS`, `HZ`를 맞추세요. `TASK`에는 녹화 때 사용한 작업 문구를, `CHECKPOINT_PATH`에는 새 모델의 `pretrained_model/` 경로를 지정합니다. [정책 추론 설정](../7_policy-inference/policy-inference.md)

### W&B에서 학습 곡선 보기 (선택)

W&B 대시보드를 사용하려면 학습 PC에서 한 번 로그인합니다.

```bash
conda run -n lerobot_act --no-capture-output wandb login
```

학습 스크립트의 `--wandb.enable=false` 줄을 아래 한 줄로 교체합니다. 학습 지표는 `vai_lab` 프로젝트에서 볼 수 있습니다. `--wandb.disable_artifact=true`는 모델 체크포인트의 W&B 업로드를 끕니다.

```bash
  --wandb.enable=true --wandb.project=vai_lab --wandb.disable_artifact=true \
```

## 3. 체크포인트 확인과 학습 재개

연구실 PC에서는 `VAI_AIWORKER/lerobot/outputs/train/`에서 이전 학습 결과를 확인할 수 있습니다. 원하는 실험 폴더의 `checkpoints/`를 열면 저장된 스텝별 모델이 있습니다.

50,000스텝 학습을 마치면 최종 모델은 실행 폴더의 `checkpoints/050000/pretrained_model/`에 저장됩니다.

```text
outputs/train/
├── act_pick_and_place_arm16_bs256_50k.log
└── act_pick_and_place_arm16_bs256_50k/checkpoints/
    ├── 002000/
    ├── 004000/
    ├── …
    ├── 048000/
    ├── 050000/
    │   ├── pretrained_model/
    │   └── training_state/
    └── last -> 050000/
```

첫 체크포인트가 저장된 뒤 학습이 중단되면 다음 스크립트로 이어갑니다. `--config_path`는 최근 체크포인트에 저장된 학습 설정 파일을 가리킵니다. `--resume=true`를 주면 모델과 `training_state/`를 복원해 원래 목표인 50,000스텝까지 학습합니다.

```bash title="resume_act_arm16.sh"
#!/bin/bash
set -euo pipefail

cd /home/vai/Desktop/yujin/VAI_AIWORKER/lerobot
RUN_DIR=outputs/train/act_pick_and_place_arm16_bs256_50k

conda run -n lerobot_act --no-capture-output lerobot-train \
  --config_path="$RUN_DIR/checkpoints/last/pretrained_model/train_config.json" \
  --resume=true \
  --wandb.enable=false \
  2>&1 | tee -a "${RUN_DIR}.log"
```

재개 스크립트도 W&B를 끄도록 지정했으므로 이전에 W&B를 켜고 저장한 체크포인트에서도 로그인 없이 이어갈 수 있습니다. 재개 후에도 대시보드에 기록하려면 위와 같이 W&B 옵션 줄을 교체하세요.

학습한 모델은 로봇 없이 합성 입력으로 점검할 수 있습니다. 추론 노트북의 `CHECKPOINT_PATH`에는 이번 실행의 `checkpoints/050000/pretrained_model` 경로를 지정하고, `EXCLUDED_JOINTS`와 `CAMERAS`는 학습 데이터의 관절·카메라 구성에 맞추세요. [정책 추론 실행](../7_policy-inference/policy-inference.md)
