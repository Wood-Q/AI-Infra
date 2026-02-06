# nano-vLLM 代码阅读指南（面向 vLLM 新手）

这份指南的目标是：**沿着一次 `LLM.generate()` 的真实执行链路**把 nano-vllm 读“通”，并在阅读过程中补齐你需要的推理/训练知识点。

> 建议阅读方式：边跑边断点。你每读完一个阶段，就能回答一个问题：**“这一阶段在解决什么性能/正确性问题？”**

---

## 0. 你要先知道的三件事

1. **推理分两段**：
   - **Prefill（也叫 prompt 阶段）**：把整段 prompt 一次性喂给模型，计算并写入 KV cache。
   - **Decode（也叫 generation 阶段）**：每步只喂一个新 token，重复利用 KV cache 快速生成。

2. **vLLM 的核心不是“模型结构”，而是“怎么把 KV cache 管好、怎么调度 batch”**。
   nano-vllm 把这些点用很短的代码复刻了：调度（Scheduler）+ 块式 KV（BlockManager / block_table）+ FlashAttention 的 paged/varlen 接口。

3. **读代码优先读“数据结构”和“状态机”**：
   - Sequence 的状态（WAITING/RUNNING/FINISHED）
   - block_table / slot_mapping 的意义
   - prefill/decode 两条路径如何共用同一套 attention 实现

---

## 1. 推荐阅读顺序（按调用链，从外到内）

### 第 1 站：先跑通用法与入口

- [example.py](example.py)
  - 你要确认：prompt 如何被 tokenizer 编码、如何批量调用 `generate`、输出结构是什么。
  - 需要补的知识：
    - 分词与 chat template（为什么 chat 模板里会有 generation prompt）。

- [nanovllm/llm.py](nanovllm/llm.py)
  - 这里基本只是把 `LLM` 暴露成 `LLMEngine`。

### 第 2 站：主循环（你会在这里看到 prefill/decode 的分界）

- [nanovllm/engine/llm_engine.py](nanovllm/engine/llm_engine.py)
  - 重点读：
    - `LLMEngine.generate()`：外层循环直到 `scheduler.is_finished()`
    - `LLMEngine.step()`：一次调度 + 一次模型运行 + 一次后处理
  - 你要能说清楚：
    - `step()` 返回的 `num_tokens` 为什么 prefill 是正数、decode 是负数（用于区分吞吐统计）。

- [nanovllm/config.py](nanovllm/config.py)
  - 重点读：最大 batch token、最大 seq 数、KV cache block 配置、TP 大小。

- [nanovllm/sampling_params.py](nanovllm/sampling_params.py)
  - 重点读：只支持 temperature sampling（禁止 greedy）。
  - 需要补的知识：
    - temperature 的数学含义：$p_i \propto \exp(\frac{logit_i}{T})$。

### 第 3 站：调度器（vLLM 的“吞吐感”从这里来）

- [nanovllm/engine/scheduler.py](nanovllm/engine/scheduler.py)
  - 重点读两条路径：
    - `schedule()` 的 **prefill 分支**：从 waiting 拉新请求、分配 block、把 seq 放入 running。
    - `schedule()` 的 **decode 分支**：从 running 里轮转，确保能 append（必要时 preempt）。
  - 你要抓住的两个点：
    - **批量约束**：`max_num_seqs` 与 `max_num_batched_tokens`
    - **抢占（preempt）**：KV cache 不够就把别的 seq 退回 waiting

- [nanovllm/engine/sequence.py](nanovllm/engine/sequence.py)
  - 重点读：
    - `num_cached_tokens` / `block_table` / `num_blocks` / `last_block_num_tokens`
    - `__getstate__` / `__setstate__`：为什么多进程传输时只发 last_token（decode）

### 第 4 站：块管理 + 前缀缓存（PagedAttention 的“paged”在这里）

- [nanovllm/engine/block_manager.py](nanovllm/engine/block_manager.py)
  - 你会看到 vLLM 的一个关键思想：**把 KV cache 切成 block，用 block_table 表示“这个序列的 KV 在哪几个 block”**。
  - 重点读：
    - `allocate()`：
      - 用 `xxhash` 对“满 block 的 token_ids”做 hash，命中就直接复用（prefix cache）。
      - miss 就从 `free_block_ids` 分配新 block。
    - `may_append()`：
      - 序列增长时，什么时候需要新 block、什么时候把 block 变成可缓存（hash != -1）。
  - 需要补的知识：
    - **KV cache 内存怎么估算**（按层数/heads/head_dim/token 数）。
    - prefix cache 的适用场景：大量请求共享相同前缀（例如系统 prompt）。

### 第 5 站：模型执行器（把“调度后的 seqs”变成一次 GPU kernel 调用）

- [nanovllm/engine/model_runner.py](nanovllm/engine/model_runner.py)
  - 这是第二个“核心文件”。建议按这个顺序读：
    1. `__init__()`：TP 进程组初始化、加载模型、warmup、分配 KV cache、（可选）捕获 CUDA graph。
    2. `allocate_kv_cache()`：根据可用显存推导 `num_kvcache_blocks`，并把每层的 `k_cache/v_cache` 指向一个大 buffer。
    3. `prepare_prefill()`：
       - 拼接所有 seq 的 input_ids/positions
       - 构建 `cu_seqlens_q/k`（变长 batch 的前缀和）
       - 构建 `slot_mapping`（把“新 token 的 K/V”写到 KV cache 的哪个 slot）
       - 如果存在 prefix cache：准备 `block_tables`
    4. `prepare_decode()`：
       - 每个 seq 只喂一个 last_token
       - 构建 `context_lens`（当前上下文长度）+ `block_tables`
    5. `run_model()`：prefill 走 eager；decode 在合适条件下走 CUDA graph replay。
    6. `run()`：执行 model→logits→sampler，并在每步后 `reset_context()`。

- [nanovllm/utils/context.py](nanovllm/utils/context.py)
  - 这是一个“简化版全局上下文”：attention kernel 通过它拿到 slot_mapping / block_tables / cu_seqlens 等。

### 第 6 站：注意力与 KV cache 写入（你会在这里看到 FlashAttention + Paged KV）

- [nanovllm/layers/attention.py](nanovllm/layers/attention.py)
  - 重点读：
    - `store_kvcache()`：Triton kernel 把新算出来的 K/V 写进 KV cache（slot_mapping 决定写到哪里）。
    - `Attention.forward()`：
      - prefill：`flash_attn_varlen_func`（变长 + causal + 可选 block_table）
      - decode：`flash_attn_with_kvcache`（直接读 KV cache 做注意力）
      - prefix cache 场景下：prefill 的 K/V 可以直接用 cache 里的（通过 block_tables 指示）。
  - 需要补的知识：
    - FlashAttention 的核心：把 attention 的中间矩阵不落显存，减少带宽瓶颈。
    - “paged” 的本质：不是连续的 KV，而是通过 block_table 间接寻址。

### 第 7 站：模型结构与权重加载（模型本身不是 vLLM 的难点，但要知道它如何适配 TP）

- [nanovllm/models/qwen3.py](nanovllm/models/qwen3.py)
  - 重点读：
    - `Qwen3Attention`：QKV 并行线性层 + RoPE + 调用 `Attention`。
    - `packed_modules_mapping`：把 HF 权重名映射到 nano-vllm 的合并权重布局。

- [nanovllm/utils/loader.py](nanovllm/utils/loader.py)
  - 重点读：
    - safetensors 遍历加载
    - 对 packed 模块走自定义 weight_loader（按 shard_id 分片/合并）

- 建议最后再扫一遍这些层，主要是理解 TP 切分方式：
  - [nanovllm/layers/linear.py](nanovllm/layers/linear.py)
  - [nanovllm/layers/embed_head.py](nanovllm/layers/embed_head.py)
  - [nanovllm/layers/rotary_embedding.py](nanovllm/layers/rotary_embedding.py)

### 第 8 站：采样（最容易改来做练习）

- [nanovllm/layers/sampler.py](nanovllm/layers/sampler.py)
  - 这里用的是一种向量化的采样写法（Gumbel-max 风格）：
    - `probs / exponential` 再 `argmax`
  - 练习方向：加 top-k / top-p，或者允许 greedy（temperature → 0 的特殊分支）。

---

## 2. “边读边补”的推理知识清单（按你在代码里会遇到的顺序）

### A. Prefill vs Decode 的计算与复杂度

- Prefill 的计算量大约是 $O(L^2)$（每层 attention 对整段长度 L 做 causal attention）。
- Decode 每步只新增 1 个 token，但要对历史 KV 做注意力，单步大约是 $O(L)$。
- 所以优化目标通常是：
  - **提升 decode 吞吐**（减少 Python 调度开销、用 CUDA graph、把 batch 做大）
  - **降低 KV cache 的显存占用/碎片**（paged/block）

### B. KV cache 的内存模型（建议你手算一次）

在 [nanovllm/engine/model_runner.py](nanovllm/engine/model_runner.py) 的 `block_bytes` 公式里，基本就是：

- $\text{KV bytes} \approx 2 \times \text{layers} \times \text{tokens} \times \text{kv_heads} \times \text{head_dim} \times \text{dtype_bytes}$

这里“2”来自 K 和 V。

### C. Paged/block KV 的关键数据结构

- `block_table`：每个序列拥有的 block id 列表
- `slot_mapping`：这次要写入的 token 的 KV，写到 KV cache 的哪个“线性 slot”
- `block_tables`：一个 batch 的二维表（补齐到同宽），给 FlashAttention 的 paged 接口用

### D. Prefix cache（前缀复用）

- 命中前提：
  - 以 block_size 为粒度（满块）
  - token 完全一致
- 适合：系统 prompt、长模板、RAG 固定前缀等。

### E. CUDA Graph（为什么 decode 常用它）

- decode 每步 shape 很固定（batch 变化有限），适合 capture → replay。
- nano-vllm 用“bucket 化的 batch size”（1/2/4/8/16...）来减少 capture 次数。

### F. Tensor Parallel（TP）在推理里怎么“切”

- 注意力里最常见的是按 head 切分：每个 rank 拿一部分 heads。
- nano-vllm 的 Qwen3Attention 明确了：`num_heads` / `num_kv_heads` 都按 `world_size` 平均分。

---

## 3. 读代码时顺便补的训练知识（不要陷太深，但要知道差异）

> vLLM/nano-vllm 主要解决“推理系统”问题；训练代码（反向、优化器、数据并行）不在这套代码里。

你读推理代码时建议同步补这些概念（理解推理为何能省这么多）：

- **Teacher forcing vs 自回归生成**：
  - 训练：一次 forward 看到整段目标序列（shifted labels），并做反向传播。
  - 推理：逐 token 生成（decode loop）。
- **训练为什么通常不用 KV cache**：
  - 训练要算梯度、activation 很多，且通常一次性算整段；KV cache 的价值在 decode 复用。
- **损失与优化**：交叉熵、label shift、学习率调度、混合精度（fp16/bf16）。
- **并行策略对比**：
  - 训练常见：DP/PP/TP/ZeRO
  - 推理常见：TP +（有时）流水/多实例；核心瓶颈是显存与带宽。

---

## 4. 建议你做的 6 个“边读边练”小实验（强烈推荐）

1. **Prefill/Decode 形状跟踪**：在 `ModelRunner.prepare_prefill/prepare_decode` 打印张量 shape，理解变长 batch 如何拼接。
2. **Prefix cache 命中验证**：构造两个 prompts 共享长前缀，观察 `seq.num_cached_tokens` 是否增加。
3. **抢占验证**：降低 `gpu_memory_utilization` 或增大并发，让 `Scheduler.preempt()` 触发，观察等待队列变化。
4. **KV cache 占用估算**：手算 `block_bytes`，再用 `torch.cuda.mem_get_info()` 对照。
5. **采样改造**：在 sampler 加 top-k（先实现最简单版本），并对输出多样性做对比。
6. **吞吐对比**：分别测试 `enforce_eager=True/False`，观察 decode tok/s 的变化（CUDA graph 的收益）。

---

## 5. 你读完应该能回答的“面试题式”问题

- 为什么 vLLM 需要 paged KV cache？它解决了什么内存问题？
- prefill 为什么可以做 varlen batch？decode 为什么要维护 block_table？
- prefix cache 什么时候能命中？为什么要以 block 为粒度？
- CUDA graph 对 decode 有效的原因是什么？它的限制是什么？
- TP 切 head 时，为什么 `num_kv_heads` 也要切？GQA/MQA 会带来什么影响？

---

## 6. 下一步怎么“从 nano 走向 vLLM”

当你把上述链路读通后，再去看 vLLM 代码会很顺：

- 把 nano-vllm 的概念映射到 vLLM：
  - Sequence/Request → SequenceGroup / Request
  - BlockManager → KVCacheManager / BlockAllocator
  - block_table/slot_mapping → PagedAttention 的 metadata
  - Scheduler 的两段逻辑 → vLLM 更复杂的调度（优先级、流式输出、多阶段执行）

如果你愿意，我也可以在你读完这份指南后，按你当前进度给你出一套“对照 vLLM 源码”的阅读路线（更偏工程系统）。
