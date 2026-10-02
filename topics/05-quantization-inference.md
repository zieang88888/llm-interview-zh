# 05 · 量化与推理部署

> 核心考点：INT8/INT4、GPTQ/AWQ、vLLM/PagedAttention、Continuous Batching、Speculative Decoding

---

### Q1 【基础】为什么需要量化？INT8 和 FP16 的区别？

- **答案要点**：
  1. 为什么量化：FP16 模型太大、显存太贵、推理慢；量化把权重从 16-bit 浮点数压到 8-bit/4-bit 整数，显存减半甚至减到 1/4，推理速度提升。
  2. FP16：每个参数 2 字节，动态范围大、精度高；是模型训练和推理的默认精度。
  3. INT8：每个参数 1 字节，显存减半，速度提升约 2 倍；精度损失很小（通常 <1% benchmark 下降）。
  4. INT4：每个参数 0.5 字节，显存降到 1/4；精度损失略大，但对大模型（>30B）影响很小，是当前主流部署方案。
- **追问点**：
  - 量化为什么对大模型影响小？（答：大模型参数冗余度高，量化相当于一种结构化剪枝，冗余参数的精度损失不影响整体输出；小模型量化掉点明显）
  - 训练时能量化吗？（答：训练一般不用 INT8/4，因为梯度需要高精度；只有推理时用量化；QLoRA 是个例外——基座 4-bit 但 LoRA 参数 FP16）

---

### Q2 【基础】GPTQ 和 AWQ 的区别？

- **答案要点**：
  1. GPTQ：一种训练后量化（PTQ）方法，逐层量化，逐列误差补偿；基于二阶信息（Hessian 矩阵）最小化量化误差；效果好但实现复杂。
  2. AWQ（Activation-aware Weight Quantization）：发现不是所有权重都同等重要——保护 1%~10% 的"显著权重"（和激活值大小相关的），其余正常量化；不需要逐层误差补偿，速度快很多。
  3. 核心区别：GPTQ 从数学上最小化量化误差；AWQ 从"哪些权重更重要"出发做保护。实际效果两者相当，AWQ 量化速度更快、推理时 kernel 优化更好。
  4. 选型经验：追求速度和易用性选 AWQ；追求极限精度选 GPTQ。
- **追问点**：
  - 为什么激活值大的权重更重要？（答：因为权重的实际贡献 = 权重 × 激活值；激活值大的通道上的权重对输出影响大，量化这些权重误差会被放大）
  - 量化后的模型能再微调吗？（答：可以，QLoRA 就是 4-bit 基座 + LoRA 微调；但全参量化微调不常见）

---

### Q3 【进阶】vLLM 的 PagedAttention 解决了什么问题？

- **答案要点**：
  1. 传统问题：KV Cache 按最大序列长度预分配连续显存，大量空间被浪费——内部碎片（预留了但没用满）+ 外部碎片（不同长度请求导致显存碎片）。
  2. PagedAttention 思想：借鉴操作系统的虚拟内存分页机制，把 KV Cache 切成固定大小的 block（如每个 block 存 16 个 token 的 KV）；逻辑上连续，物理上离散；通过 block table 映射。
  3. 效果：显存利用率从 ~20% 提升到 ~90%；可以并发处理更多请求；吞吐量提升 2~4 倍。
  4. 额外优势：不同请求之间可以共享 KV Cache（如 parallel sampling 同一个 prompt 生成多个输出）。
- **追问点**：
  - vLLM 还有什么关键特性？（答：Continuous Batching、PagedAttention、量化支持、张量并行、流式输出；是目前最流行的开源推理引擎）
  - PagedAttention 有什么缺点？（答：分页映射有额外开销；对很短的序列反而 overhead 大）

---

### Q4 【实战】推理时 OOM 了，你有哪些优化手段？

- **答案要点**：
  1. 降低模型精度：FP16 → INT8 → INT4，显存直接减半/减到 1/4。
  2. 减少 KV Cache 占用：用 GQA/MQA 模型；KV Cache 量化到 INT8；减少 max_batch_size；缩短 max_seq_len。
  3. 减小 batch size：最简单直接，但会降低吞吐。
  4. 模型并行：多卡张量并行（TP）或流水线并行（PP），把模型切到多张卡上。
  5. 用更好的推理框架：vLLM/TensorRT-LLM 做 PagedAttention + Continuous Batching，显存利用率高。
  6. 换小模型：如果 70B 跑不动，考虑 34B 或 13B 模型，效果差距可能不大。
- **追问点**：
  - 7B 模型 FP16 推理大概要多少显存？（答：模型权重 ~14GB；KV Cache 看并发和上下文长度；单请求短上下文 ~16GB 够用，长上下文多并发需要 24GB+）
  - 怎么估算推理显存？（答：模型权重 + KV Cache + 激活值 + 框架 overhead；KV Cache 是大头，和并发数×上下文长度成正比）

---

### Q5 【进阶】Continuous Batching 是什么？和 Static Batching 的区别？

- **答案要点**：
  1. Static Batching：等一批请求凑齐后一起跑，跑完这一批再跑下一批；问题是长请求要等短请求，GPU 利用率低。
  2. Continuous Batching（也叫 In-flight Batching）：不等整批完成——一个请求生成完了，立刻从队列里取下一个新请求插进来；动态拼 batch。
  3. 效果：GPU 利用率从 ~20% 提升到 ~60%+；吞吐量提升 3~8 倍。
  4. 为什么之前不这么做？（答：传统训练框架是静态 batch 设计；推理场景下 vLLM 等重新设计了调度器，支持动态拼接 batch）
- **追问点**：
  - Continuous Batching 为什么能提升吞吐？（答：短请求不用等长请求，GPU 永远有事做；避免了 batch 中最慢的请求拖慢整批的问题）

---

### Q6 【实战】首 token 延迟（TTFT）和吞吐量怎么优化？

- **答案要点**：
  1. TTFT（Time To First Token）：用户发请求到收到第一个 token 的时间；主要瓶颈是 Prefill 阶段（处理整个 prompt 的大矩阵乘）。
  2. 优化 TTFT：(a) 用更快的 Prefill 实现（FlashAttention）；(b) Chunked Prefill：把长 prompt 的 prefill 拆成小块，和 decode 请求混合调度，不阻塞 decode；(c) 减少 prompt 长度。
  3. 吞吐量优化：(a) Continuous Batching 提高 GPU 利用率；(b) PagedAttention 减少显存碎片；(c) 量化降低显存、提高 batch size；(d) Speculative Decoding。
  4. 权衡：TTFT 和吞吐量有时候是 trade-off——batch 越大吞吐越高，但每个请求的 TTFT 可能增加。
- **追问点**：
  - Prefill 和 Decode 为什么要分开优化？（答：Prefill 是计算密集型（大矩阵乘，GPU 算力吃得满），Decode 是内存密集型（逐 token 读 KV Cache，算力利用率低）；二者最优 batch 策略不同）

---

### Q7 【进阶】Speculative Decoding 的原理是什么？

- **答案要点**：
  1. 问题：大模型 Decode 阶段逐 token 生成，每步都要跑一次完整 forward pass，内存带宽是瓶颈。
  2. 思路：用一个小的"草稿模型"（draft model）快速生成 k 个候选 token；然后大模型一次并行验证这 k 个 token 是否合理；接受匹配的部分，拒绝的位置重新采样。
  3. 效果：如果草稿模型质量不错（接受率高），可以把 Decode 速度提升 2~3 倍，且输出分布和大模型完全一致（无损）。
  4. 变体：Medusa（在大模型上加多个头并行预测多个 token）、EAGLE（用特征层做草稿）。
- **追问点**：
  - Speculative Decoding 会改变输出结果吗？（答：不会——验证阶段保证最终输出和大模型逐 token 采样的分布完全一致，是无损加速）
  - 草稿模型怎么选？（答：一般选同系列的小模型，如 7B 作为 70B 的草稿；接受率和草稿模型质量正相关）

---

### Q8 【实战】部署 70B 模型，显存不够，你会怎么选型和部署？

- **答案要点**：
  1. 显存估算：70B FP16 权重 ~140GB；INT4 量化后 ~35GB。
  2. 单卡方案：INT4 量化 + vLLM，单张 A100 80GB 或 H100 80GB 可以跑，但 KV Cache 空间有限，并发不能太高。
  3. 多卡方案：张量并行 TP=2 或 TP=4，每卡放一部分权重；FP16 部署需要 2~4 张 80GB 卡。
  4. 推荐方案：如果追求性价比 → INT4 量化 + 单/双卡 vLLM；如果追求效果和并发 → FP16 + 4 卡 TP。
  5. 备选：考虑用 34B/32B 模型，INT4 后单卡就能跑，效果和 70B 差距不大，成本低很多。
- **追问点**：
  - 张量并行和流水线并行的区别？（答：TP 是层内切分（每个矩阵拆到多卡），通信频繁但延迟低，适合单节点多卡；PP 是按层切分（不同层放不同卡），通信少但有 bubbles，适合多节点）

---

### Q9 【进阶】MoE 模型推理有什么特殊挑战？

- **答案要点**：
  1. MoE（Mixture of Experts）：每次只激活少数几个 expert（如 8 个 expert 选 2 个）；总参数很大但激活参数少。
  2. 推理挑战：
     - (a) 专家路由不均衡：不同请求激活不同专家，GPU 显存里要放所有专家的权重，但每次只用一小部分，显存利用率低。
     - (b) All-to-All 通信：token 要路由到对应专家所在的卡，多卡时通信开销大。
     - (c) 动态 batch：不同 expert 的 batch 大小不一样，调度复杂。
  3. 优化：Expert Parallelism（EP）——每个卡放几个 expert，token 按路由结果卡间传输；专家量化降低显存。
- **追问点**：
  - MoE 模型推理为什么比同 dense 模型慢？（答：虽然激活参数少，但 all-to-all 通信开销大；且专家分散在多卡，每次都要跨卡传输 token）

---

### Q10 【实战】怎么选推理框架？vLLM / TensorRT-LLM / Ollama 各适合什么场景？

- **答案要点**：
  1. vLLM：生产级、高吞吐、支持 Continuous Batching + PagedAttention；适合在线服务、高并发场景；Python 生态友好。
  2. TensorRT-LLM：NVIDIA 官方、极致优化（kernel 级优化）；延迟最低、吞吐最高，但部署复杂、灵活性差；适合延迟敏感的在线场景。
  3. Ollama：本地部署、一键安装、开发者友好；适合个人本地使用、原型验证；不适合高并发生产环境。
  4. 其他：SGLang（结构化生成优化）、Text Generation Inference（HuggingFace 出品）、llama.cpp（CPU/边缘端）。
- **追问点**：
  - 生产环境一般选什么？（答：大部分团队选 vLLM——平衡了性能和易用性；极致性能团队会用 TensorRT-LLM）

---
