# 视频 LoRA Fleet 压缩实验计划（Remade-AI × Wan2.1-I2V-14B-480P）

更新：2026-10-01。目标：在同一个底模上常驻一整个 LoRA fleet，用 FlashTSQR 做
fleet 级全局谱分配压缩，验证能否在保持生成质量的同时显著减少 adapter 显存。

## 已完成（Arrhenius）

| 步骤 | 结果 | 位置 |
| --- | --- | --- |
| 1. HF 同底模 LoRA 普查（只读 header） | 2,171 个文件；Wan2.1-I2V-480P 有 103 个仓库结构一致 | [README.md](README.md)，`results/fleet_candidates_login-20261001.json` |
| 2. 下载 fleet | Remade-AI 49 个特效 LoRA，rank 32×480 模块，BF16，17.60 GB，sha256 全部校验 | `results/remade_wan21_i2v_480p_manifest.json` |
| 3. 下载底模 | Comfy-Org BF16 重打包（DiT 32.8 GB + umT5 11.4 + CLIP 1.3 + VAE 0.25），sha256 校验 | `results/wan21_i2v_480p_comfy_bf16_fetch.sh` |
| 4. FlashTSQR 全 fleet 谱（作业 3237886） | 23,520 模块，谱计算 64 s（GH200） | `results/fleet_spectrum_3237886/` |
| 5. 压缩冒烟 | Rotate b50：7,516/15,360 方向，能量 0.977，359→183 MB，3.6 s | 作业 3239681 日志 |

谱分配结果（全局 top-k，按每个 adapter 总能量归一化）：

| 预算 | 紧凑 BF16 | 单 adapter 能量 最差 / 中位 |
| --- | ---: | --- |
| 原始 | 17.60 GB | 1 / 1 |
| b75 | 13.53 GB | 0.996 / 0.997 |
| b50 | 9.14 GB | 0.974 / 0.978 |
| b25 | 4.84 GB | 0.912 / 0.921 |

能量只是代理指标，**生成质量尚未验证**——这是下一步。

## 待做

### A. 试点生成（阻塞在 Arrhenius GPU 排队；作业 3242843 冒烟、3243012 全量）

- adapter：Assassin（b50 能量最差）、Rotate（下载最多）、angry-face（能量最好）。
- 每个 adapter：无 LoRA 基线 + 原始 + b75 + b50 + b25，共 15 段视频。
- 设置：Wan 官方示例图（戴墨镜的猫），480×832，49 帧，30 步，CFG 6，flow shift 5，
  seed 42；提示词见 `configs/remade_pilot_prompts.json`（Remade 模板 + 触发词）。
- LoRA 以 diffusers/PEFT 非合并方式加载，逐个 load/delete。
- 指标：对原始版的 PSNR；偏离比 ‖v−orig‖/‖base−orig‖（0=与原始相同，1=和不挂 LoRA 一样远）；
  每段的峰值显存、adapter 实际 GPU 字节、耗时。
- 判断标准（建议）：b50 偏离比明显 < 0.3 且肉眼特效保留 → 进入 B；否则看 b75。
- 冒烟用 `STEPS=2 FRAMES=9 ADAPTERS=Rotate BUDGETS=0.5` 先验证整条链路
  （CLIP `position_ids` 问题已修，但修复后尚未在节点上跑过）。

### B. 全 fleet 质量（A 通过后）

- 49 个 adapter × {原始, 选定预算}，每个 2 个种子；如时间允许加入 LPIPS/CLIP 分数。
- 报告每个 adapter 的偏离比，找出最差的长尾。

### C. 真实显存（关键卖点）

- 同一进程内常驻 N 个 adapter（N = 1, 8, 16, 49），非合并，记录 `memory_allocated`。
- 原始 vs 紧凑：预期 49 个时约 17.6 → 9.1 GB（b50），需实测，零填充不算。
- 另测 adapter 切换延迟与单段生成延迟，区分合并 / 非合并路径。

### D. 扩展（可选）

- 同结构的其余约 54 个 I2V-480P 仓库（需内容筛查），共 39.5 GB。
- 与逐 adapter 均匀截断（每个都留 50%）对比，证明全局分配的价值。

## 在另一台机器上复现

需要：CUDA GPU（试点约 50–60 GB 显存）、`nvcc`、Python 环境含
torch、diffusers≥0.35、transformers、peft、safetensors（脚本会自动装 imageio、imageio-ffmpeg、ftfy）。
所有数据路径都通过环境变量传入；`PROBE_SCRATCH` 指定编译/临时目录。

```bash
# 1) fleet（17.6 GB，按 manifest 固定版本+sha256 下载）
python tools/fetch_lora_fleet.py \
  --manifest docs/video_lora_fleet_survey/results/remade_wan21_i2v_480p_manifest.json \
  --output-dir $DATA/video_lora_fleets/remade_wan21_i2v_480p

# 2) 底模：照 results/wan21_i2v_480p_comfy_bf16_fetch.sh 下载（改其中 out 路径），
#    仓库 Comfy-Org/Wan_2.1_ComfyUI_repackaged@123acf1；
#    另需 Wan-AI/Wan2.1-I2V-14B-480P-Diffusers@b184e23 的配置/tokenizer/示例图
#    放到 $MODEL_DIR/diffusers_config/（文件清单见下）
#    model_index.json image_encoder/config.json image_processor/preprocessor_config.json
#    scheduler/scheduler_config.json text_encoder/config.json tokenizer/* transformer/config.json
#    vae/config.json examples/i2v_input.JPG

# 3) 试点（谱结果已随仓库提交，不必重算）
export PROBE_PYTHON=... PROBE_CUDA_HOME=... PROBE_SCRATCH=... \
       FLEET_DIR=$DATA/video_lora_fleets/remade_wan21_i2v_480p MODEL_DIR=... \
       SPECTRUM_RUN=$PWD/docs/video_lora_fleet_survey/results/fleet_spectrum_3237886
STEPS=2 FRAMES=9 ADAPTERS=Rotate BUDGETS=0.5 bash tools/wan_fleet_pilot.sbatch   # 冒烟
bash tools/wan_fleet_pilot.sbatch                                                # 全量
```

无 Slurm 时直接 `bash` 运行：需手动设 `SLURM_SUBMIT_DIR=$PWD SLURM_JOB_ID=<run-id>`，
并去掉/替换脚本里的 `source ~/.config/hpc-agent/bootstrap.sh`。
重算谱：`tools/fleet_spectrum_flash.py --fleet-dir $FLEET_DIR --output-dir <新目录>`。

## 注意事项

- 压缩必须走 FlashTSQR（`compress_fleet_flash.py` 调用 `vla_fleet/compress_oft_gpu.truncate`），
  并检查奇异值与谱运行一致；日志需出现 `compression backend: FlashTSQR`。
- Remade 文件无 `alpha` 张量 → 缩放为 1；压缩输出同样不写 alpha，保持一致。
- 底模是 Comfy-Org 重打包，不是 Wan 官方发布文件，报告时注明。
- 谱能量 ≠ 生成质量；字节节省只在非合并、紧凑存储时成立。
- Arrhenius 存储：bloom 项目仍超软配额（250 GB），宽限期至约 2026-10-25。
