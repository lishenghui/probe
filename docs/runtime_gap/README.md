# PROBE 压缩收益为何没有充分转成推理与显存收益

结论：目前同时存在两种情况。历史质量评估路径只做数学截断、保留零填充形状；
真正紧凑导出可以兑现 adapter 字节节省，但这三个低 rank fleet 的 adapter
在整模型内存中占比很小，而且 rank 分配没有直接优化实际执行成本。
现有证据支持压缩收益，尚不支持普遍或大幅的端到端推理加速。

本次没有修改历史结果、重新压缩模型、下载权重或重新测量整模型吞吐。
新证据是 GH200 上的 factor-only 内存实测、历史分配的字节/结构核算和旧计时审计。
新测量作业 `3222950` 完成，退出码 `0:0`，用时 38 秒。
首次作业 `3222936` 因历史 validation JSON 的嵌套结构解析错误失败；修正解析后重跑。

## 1. 数学上的低 rank 不等于物理压缩

[`compress_adapter.py`](../../ruller-paper/experiments/rq3/compress_adapter.py)
明确保留原始 A/B shape，`truncated_factors()` 用 `zeros_like` 创建同尺寸张量，
仅填入前 k 个方向，配置文件也原样复制。这个设计便于保持原 scaling、验证质量，
但普通 dense kernel 仍处理原 rank，权重存储也保留零元素。
其中的 `lora_parameter_fraction_if_repacked` 本身就表示“如果重新打包后的比例”。

[`compact_adapter.py`](../../loraforge_kernels/compact_adapter.py) 已提供正确的部署变换：
裁成真实逐层 rank，匹配 `rank_pattern` 与 `alpha_pattern`，保持 scaling，
rank=0 时删除 target。无需重新发明这个 exporter，但必须实际接到部署入口。
它只支持其检查过的普通 Linear LoRA/rsLoRA，不能假定任意 adapter 类型都适用。

新 GPU 检查从保留的 compact 权重重建 nominal-rank 零填充表示，统一按 BF16
分配到 GH200。没有使用 QR/SVD，也没有进行新压缩；因此不是 FlashTSQR 运行结果。

| 真实 adapter | 零填充 A/B | 紧凑 A/B | PyTorch allocated 增量 | 紧凑时 reserved 增量 | 非零分支 |
|---|---:|---:|---:|---:|---:|
| ARC Challenge | 8 MiB | 1.5 MiB | 1.5 MiB | 2 MiB | 64 → 64 |
| Story Cloze | 8 MiB | 0.125 MiB | 0.125 MiB | 2 MiB | 64 → 8 |
| SST-2 | 8 MiB | 2.625 MiB | 2.625 MiB | 4 MiB | 64 → 64 |

保留的磁盘权重为 FP32，上表是显式转换为 BF16 后的 GPU 张量字节。
每种表示测量前释放上一组张量并清理 allocator 缓存。
这没有加载 base、PEFT、KV cache 或 CUDA Graph，不是整进程显存测量。
`allocated` 和 `reserved` 的差异在这个小实验里已清楚可见。
PyTorch 的缓存仍可能显示在 `nvidia-smi`，不能只凭后者判断压缩是否生效。
参见 [PyTorch 内存说明](https://docs.pytorch.org/docs/stable/notes/cuda.html#memory-management)。

## 2. 当前三个 fleet 很难产生很大的整模型内存降幅

重新读取 `fra_clean_alloc/*_fra_b*.json` 的逐层分配，按每个 target 的维度计数，
而非直接用 rank 比例代替参数比例。Mistral q 每个方向是 8192 个参数，k/v 是
5120；Llama-2 q/v 都是 8192。模块顺序来自对应 functional curve，并对齐检查。
以下为 binding 分配，预算分别为 1321/8811/3191。

| Fleet | N / 原 rank | 原 A/B，BF16 | FRA 紧凑 A/B，BF16 | 节省 | 占“7B base＋全 fleet”的降幅 |
|---|---:|---:|---:|---:|---:|
| LoRA Land | 12 / 8 | 78.00 MiB | 16.60 MiB | 61.40 MiB | 0.46% |
| Lots-of-LoRAs | 25 / 16 | 450.00 MiB | 107.39 MiB | 342.61 MiB | 2.48% |
| LoRARetriever | 41 / 8 | 328.00 MiB | 49.75 MiB | 278.25 MiB | 2.03% |

最后一列是假定 7,000,000,000 个 BF16 base 参数、该 fleet 全部常驻的估算，
不是这些具体基础模型的实测总显存。每个 fleet 独立计算，不把不同 base 混为一个。
加入不变的 KV cache、workspace、激活后，节省占比还会下降。只常驻少数 adapters
时，绝对节省也会更小。量化 base 会改变占比，但应作为不同部署配置单独比较。

这不否定 adapter 层面约 76%–85% 的参数节省，也不否定临近 OOM 边界时数百 MiB
的价值；它否定的是从这个比例直接推导“整模型显存减少七八成”。
新的显存结果应分别报告：磁盘 bytes、实际加载 dtype/numel、GPU allocated、
reserved、固定 KV/graph 配置下的峰值，以及内存限制下真正增加的并发/常驻容量。

## 3. 原始计时表明：删方向和删分支的效果不同

直接审计旧 `heterogeneous-fra0/measured-b3191/{timings,summary,validation}.json`。
下表是 GH200、Llama-2-7B、batch=4、输入128、输出32、Graph decode、五次重复的
历史中位数。decode 生成后续31个 token；总吞吐也包含 eager prefill。
输入是固定随机 token IDs，不能作为下游任务质量结果。

| Adapter | 参数减少 | 活跃分支 | decode：原始 → compact | 总吞吐：原始 → compact |
|---|---:|---:|---:|---:|
| ARC Challenge | 81.25% | 64 → 64 | 252.43 → 251.91 ms | 431.66 → 423.47 tok/s |
| Story Cloze | 98.44% | 64 → 8 | 252.08 → 233.90 ms | 433.70 → 480.17 tok/s |
| SST-2 | 67.19% | 64 → 64 | 252.44 → 251.20 ms | 432.22 → 432.57 tok/s |

ARC/SST-2 的普通 compact PEFT decode 改善仅约 0.21%/0.49%。Story Cloze 的
decode 延迟降低约7.21%，总吞吐增加约10.72%。这些是同一旧实验的比较；小差异
未给出置信区间，不能解释成稳定系统提升。prefill 波动还能抵消 decode 收益。

新内存检查用的是后来重建的 compact 文件。历史文档明确记录过：原计时对应的
部分权重未持久保存，重建文件并非逐字节等于原计时文件。因此这里没有把旧速度
标成新权重的速度；两组证据通过形状/字节结构关联，数值 provenance 仍分别保留。

## 4. 混合 batch 会进一步削弱“删分支”的收益

本次对三个 binding 分配重新计算了 support：某 adapter 某层 rank=0 即为零分支。
若从 N 个 adapters 中均匀、不放回抽取 D 个不同身份，一个含 z 个零分支的层
全部为零的概率是 `C(z,D)/C(N,D)`。这是结构期望，不是请求流量或吞吐实测。

| Fleet | D=1 时可跳过整层的期望比例 | D=4 时 | D=8 时 |
|---|---:|---:|---:|
| LoRA Land | 14.97% | 0.5303% | 0.0284% |
| Lots-of-LoRAs | 18.79% | 0.2724% | 0.0018% |
| LoRARetriever | 12.23% | 0.3196% | 0.0107% |

各 adapter 删掉的层不一致。混合 batch 几乎每层仍有非零请求，因此大部分层的
kernel launch 无法整体取消。逐行 rank=0 仍可能节省部分算术和访存；上表没有
否认这种收益，也不能直接预测其速度。

现有 [`ragged_branch.py`](../../loraforge_kernels/ragged_branch.py) 能按行跳过
rank=0 算术，但每个非全零模块仍初始化中间张量、启动 shrink/expand；expand
即便面对零 rank 行也读取并复制全宽 base 输出。它的 BR 是 `next_power_of_2(rmax)`，
没有强行把所有小 rank 补到16，不能把其他旧 kernel 的最小tile结论套在这里。
`rmax` 仍影响执行宽度与临时 buffer，紧凑存储和实际执行宽度是两个指标。

三个 binding 分配的全局最大 rank 仍是8/16/8。如果部署后端按最大 rank 预留
槽位，平均 rank 降低未必会改变槽位大小。尚未对当前 vLLM 实例进行实测；这是
部署检查项。官方明确说明 `max_lora_rank` 会影响内存分配，见
[vLLM LoRA 配置](https://docs.vllm.ai/en/stable/features/lora/#configuring-max_lora_rank)。
还需检查 PEFT 是否自动把 adapter 升到 FP32；旧异构 benchmark 已明确禁用这个
转换，不应把它误诊为该次 benchmark 的问题。见
[PEFT dtype 说明](https://huggingface.co/docs/peft/en/developer_guides/troubleshooting)。

## 5. 算法优化的资源与部署付费的资源没有完全对齐

当前 `two_level_allocation.py` 主要在 rank 预算下优化风险；Mistral 中同一个 rank
方向的字节成本已经不同。部署还要付 base/attention/KV 的成本，非零分支固定
开销、全宽输出访存、执行块宽度、路由及调度成本。这些都不是简单的 `sum(rank)`。

可以用诊断式 `T ≈ T_base+KV + sum(nonzero_branch_overhead + rank_dependent_work)`
理解这一差距；这不是已拟合的性能模型。只有后一项的一部分随压缩降低。
FlashTSQR 降低的是压缩准备成本，不能直接记为模型推理收益。

如果部署先 merge 成 `W + BA`，最终 dense W 的尺寸保持不变。减少 BA 的 rank
可以减少准备/传输成本，但不会缩小合并后 dense 模型或改变同形状 GEMM 工作量。
Merge 是固定单 adapter 的合理对照，动态多 adapter 则还要计算切换/恢复开销。
参见 [PEFT merge 说明](https://huggingface.co/docs/peft/main/en/conceptual_guides/lora)。

## 6. 下一轮该验证什么

优先补齐部署核验，再决定是否投入新的 kernel 或分配器：

1. 固定真实 adapter、质量目标、精度、base、KV容量、batch及输入输出长度，比较
   original、padded、compact、compact+runtime 四种表示。把实际模块形状、dtype、
   branch count、最大rank、存储bytes写进结果。相同 executor 比较压缩增益，
   相同权重比较 runtime 增益；保留无LoRA base参照。
2. 若要提升延迟，分配器需要考虑实测 `cost(layer, rank, phase, batch, adapter_mix)`，
   特别是能否在质量约束下删除更多整分支或降低实际执行宽度。不能仅把rank减小
   当成速度代理，也不能忽略混合请求对 support 并集的影响。
3. 先用少量真实 adapters，在**相同 held-out 质量和实际bytes约束**下，判断新方案
   能否显著减少分支/执行成本。联合删除必须端到端复核，不能只相信逐层损失相加。
   单 adapter / 混合 adapter、prefill / decode 分开报告，并交错测量重复试验。
4. 若目标是内存，把评估放在真实容量边界：增加了多少可常驻 adapters、KV tokens
   或满足延迟约束的并发。也可检查冷加载/换入时间，但必须实际测量，不能从
   文件缩小比例直接推导。不要人为扩大或复制fleet来制造瓶颈。

在这些核验前，更准确的定位是“质量约束下的 adapter 压缩”，不能承诺普遍显著
提升推理吞吐。已有数据还不足以证明支撑端到端收益的执行优化目标和 workload。

## 复现与证据

- [结构、内存和历史计时审计结果](audit_3222950.json)：包含输入 SHA256。
- [完成作业日志](job_3222950.out.txt)。
- [审计脚本](../../tools/audit_runtime_gap.py) 和 [Slurm 脚本](../../tools/audit_runtime_gap.sbatch)。
- 原始旧计时位于相邻 LoRAForge 的 `artifacts/kernel_bench/heterogeneous-fra0/`，
  原始分配使用 probe 的独立 `baselines/legacy/results/` 快照。

从 probe 根目录提交，先核验账户和解释器，再设置 `PROBE_PYTHON`、`PROBE_SOURCE`：

```bash
source ~/.config/hpc-agent/bootstrap.sh
sacctmgr -n show assoc where user=$USER format=account%40,partition%20
sbatch -A "$SLURM_ACCOUNT" -p "$SLURM_GPU_PARTITION" --gres="$SLURM_GPU_GRES" \
  -o "$LORAM_LOG_DIR/%x-%j.out" -e "$LORAM_LOG_DIR/%x-%j.err" \
  tools/audit_runtime_gap.sbatch
```

本次使用旧异构 benchmark 指定的现存 `conda_envs/hfedlora/bin/python`，
实际 PyTorch 为 `2.12.1+cu126`。没有安装环境或包。
