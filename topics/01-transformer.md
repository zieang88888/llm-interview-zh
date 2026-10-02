# 01 · Transformer 架构与注意力机制

> 核心考点：Self-Attention 公式、多头机制、位置编码、KV Cache、FlashAttention、RoPE、GQA/MQA

---

### Q1 【基础】Self-Attention 的计算公式是什么？为什么要除以 √d_k？

- **答案要点**：
  1. 计算公式：`Attention(Q, K, V) = softmax(QK^T / √d_k) · V`，其中 Q、K、V 均由输入 embedding 经线性变换得到。
  2. 除以 √d_k（缩放因子）的原因：当 d_k 较大时，Q 和 K 的点积结果方差会增大，导致 softmax 进入梯度极小的饱和区；除以 √d_k 将点积方差缩回到 1 附近，保证梯度流动正常。
  3. 直观理解：√d_k 归一化让不同维度的点积尺度一致，避免某些维度主导注意力权重。
- **追问点**：
  - 如果不除以 √d_k，训练时会出现什么具体问题？（答：softmax 输出趋近 one-hot，梯度趋近零，模型几乎不更新）
  - 为什么是 √d_k 而不是 d_k？（答：假设 Q、K 各维度独立同分布均值为 0 方差为 1，点积 Q·K = Σ q_i·k_i 的方差为 d_k，标准差为 √d_k，除以 √d_k 正好归一化到单位方差）

---

### Q2 【基础】Multi-Head Attention 为什么要做多头？头数一般怎么选？

- **答案要点**：
  1. 多头的作用：将 Q/K/V 投影到多个低维子空间，每个头独立做注意力，再拼接投影回原维度；不同头可以关注不同类型的依赖关系（如语法头、语义头）。
  2. 计算方式：`MultiHead(Q,K,V) = Concat(head_1, ..., head_h) · W_O`，其中 `head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)`。
  3. 头数选择经验：通常 head_dim = d_model / h 保持在 64~128 之间；如 7B 模型 d_model=4096，头数=32，head_dim=128。头数过多会导致每个头维度太小，表达能力下降。
- **追问点**：
  - 头数和模型参数量的关系？（答：W^Q/W^K/W^V/W^O 四个投影矩阵，参数量 = 4·d_model²，与头数无关，只是拆分方式不同）
  - 不同头学到的东西真的不一样吗？（答：有研究表明部分头确实 specialize 在语法/共指等不同任务上，但也有不少头功能冗余）

---

### Q3 【基础】位置编码有哪些主要方式？为什么 Transformer 需要位置编码？

- **答案要点**：
  1. 为什么需要：Self-Attention 本身是排列不变的（permutation invariant），打乱输入顺序结果不变；而语言中语序至关重要，因此必须显式注入位置信息。
  2. 正弦位置编码（原版 Transformer）：`PE(pos, 2i) = sin(pos / 10000^(2i/d_model))`，`PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))`；优点是可外推到更长序列，无需学习。
  3. 可学习位置编码（BERT/GPT-2）：直接训练一个位置 embedding 表，长度上限固定为训练时的最大长度。
  4. RoPE（旋转位置编码，LLaMA/Qwen 系列采用）：通过对 Q/K 施加旋转矩阵来注入位置信息，相对位置编码性质更好，外推能力更强。
  5. ALiBi：不加位置编码向量，而是在注意力分数上根据距离加惩罚项，外推性极佳。
- **追问点**：
  - RoPE 为什么比正弦编码效果好？（答：RoPE 编码在 Q·K 点积中天然编码了相对位置关系，即 q_m · k_n 的结果只依赖于 m-n 的相对距离，而不是绝对位置）
  - 为什么大模型很少用可学习位置编码了？（答：外推性差，训练时 4k 长度推理时 8k 就崩；RoPE + NTK/插值方案可以较好外推）

---

### Q4 【进阶】KV Cache 的原理是什么？为什么能加速自回归推理？

- **答案要点**：
  1. 原理：自回归生成时，每生成一个新 token，需要对前面所有 token 做注意力。但前面 token 的 K、V 向量在之前的步骤中已经算过了，不需要重复计算——把它们缓存起来，新 token 只算自己的 Q 去和缓存的 K/V 做注意力即可。
  2. 加速效果：Prefill 阶段（处理输入 prompt）仍然是 O(n²)，但 Decode 阶段每步从 O(n²) 降到 O(n)（只需算新 token 的 Q 和已有 K/V 的注意力）。
  3. 显存代价：KV Cache 显存 = 2 × n_layers × n_kv_heads × seq_len × head_dim × batch_size × dtype_bytes；这是长上下文推理显存的主要开销之一。
- **追问点**：
  - KV Cache 的显存占用怎么优化？（答：GQA/MQA 减少 KV 头数、KV Cache 量化（INT8 KV）、PagedAttention 减少碎片、滑动窗口只保留最近 N 层）
  - 为什么 Prefill 和 Decode 要分开调度？（答：Prefill 是计算密集型（大矩阵乘），Decode 是内存密集型（逐 token 读 KV Cache），二者 batch 策略不同，vLLM 等框架做 chunked prefill 混合调度）

---

### Q5 【进阶】FlashAttention 的核心思想是什么？为什么比标准注意力快？

- **答案要点**：
  1. 核心问题：标准注意力需要把完整的 n×n 注意力矩阵写入 HBM（显存）再读回来，这是 IO 瓶颈——计算量是 O(n²d)，但 IO 量更大，GPU 算力利用率很低。
  2. FlashAttention 做法：分块（tiling）计算，把 Q/K/V 切成小块加载到 SRAM（片上高速缓存），在 SRAM 内完成局部注意力计算并直接更新输出矩阵，不把完整的 n×n 矩阵写回 HBM。
  3. 关键技巧：online softmax（分块时动态维护 max 和 sum，不需要等全量数据再算 softmax）；反向传播时不存储注意力矩阵，重计算（recomputation）节省显存。
  4. 效果：速度提升 2~4 倍，显存从 O(n²) 降到 O(n)，支持更长上下文；数学结果和标准注意力完全一致（精确注意力，不是近似）。
- **追问点**：
  - FlashAttention 是近似算法吗？（答：不是，是精确注意力的 IO 感知重排，数学结果与标准实现逐位对齐）
  - FlashAttention v2 相比 v1 改进了什么？（答：更好的并行化策略（batch×head 并行）、减少 GPU 同步开销、更好地利用 tensor core，速度再提 2 倍左右）

---

### Q6 【进阶】Transformer 编码器和解码器的区别？BERT 和 GPT 分别用哪种？

- **答案要点**：
  1. 编码器（Encoder）：双向注意力，每个位置可以看到左右所有位置；用于理解类任务（分类、NER、嵌入）。代表：BERT。
  2. 解码器（Decoder）：带 causal mask（下三角矩阵），每个位置只能看到自己及左边的位置；用于自回归生成。代表：GPT 系列。
  3. Encoder-Decoder（原始 Transformer 论文）：编码器双向 + 解码器带 mask 并 cross-attend 编码器输出；用于 seq2seq 任务（翻译）。代表：T5、BART。
  4. 为什么现在主流 LLM 都是 Decoder-only？答：Decoder-only 架构更统一，预训练-微调流程简单，且 scaling law 表现更好；Encoder-Decoder 在长文本生成上效率不如 Decoder-only。
- **追问点**：
  - Encoder-Decoder 现在还有用武之地吗？（答：翻译、语音识别等严格 seq2seq 场景仍有使用，如 Whisper 就是 encoder-decoder）
  - Decoder-only 做理解任务效果差吗？（答：不差，经过指令微调的 Decoder-only 模型在分类/抽取等任务上完全可以对标 BERT，只是嵌入质量略逊于专门 encoder）

---

### Q7 【进阶】GQA、MQA、MHA 的区别？为什么现在主流大模型都用 GQA？

- **答案要点**：
  1. MHA（Multi-Head Attention）：Q、K、V 头数相同，每个 Q 头配独立的 K/V 头；参数量最大，效果最好。
  2. MQA（Multi-Query Attention）：所有 Q 头共享同一组 K/V 头（K/V 头数=1）；大幅减少 KV Cache 显存，推理加速明显，但效果略有下降。
  3. GQA（Grouped-Query Attention）：折中方案，将 Q 头分为若干组，每组共享一组 K/V 头；如 32 个 Q 头配 8 个 KV 头，每组 4 个 Q 头共享 K/V。
  4. 为什么主流选 GQA：KV Cache 显存大幅降低（相比 MHA 减少到 1/4 或 1/8），效果几乎无损（论文证明接近 MHA），推理吞吐显著提升。LLaMA-2/3、Qwen、Mistral 均采用 GQA。
- **追问点**：
  - GQA 训练时需要注意什么？（答：如果从 MHA 模型转 GQA，可以将同组的 K/V 头平均融合做 warm start，减少训练成本）
  - MQA 为什么效果会下降？（答：共享 K/V 头导致表达能力受限，多语义空间被压缩到同一组 K/V 中）

---

### Q8 【进阶】Pre-LN 和 Post-LN 的区别？为什么大模型普遍用 Pre-LN？

- **答案要点**：
  1. Post-LN（原版 Transformer）：`x = LayerNorm(x + Sublayer(x))`，先残差连接再做 LayerNorm；训练需要 warmup，深层网络容易梯度不稳定。
  2. Pre-LN（GPT-2 及之后）：`x = x + Sublayer(LayerNorm(x))`，先做 LayerNorm 再进子层，残差通路是干净的；梯度可以直接回传，训练更稳定。
  3. 为什么 Pre-LN 更稳定：残差支路没有被 LayerNorm 截断，梯度可以无损地从顶层回传到底层，类似 ResNet 的 identity path；不需要 warmup 也能稳定训练。
  4. Pre-LN 的缺点：最终表现可能略逊于 Post-LN（有论文指出 Pre-LN 的表示上界略低），但工程稳定性远优先。
- **追问点**：
  - RMSNorm 是什么？和 LayerNorm 区别？（答：RMSNorm 去掉了均值中心化和偏置项，只做 RMS 归一化，计算更快，效果相当；LLaMA/Qwen 都用 RMSNorm）

---

### Q9 【进阶】RoPE（旋转位置编码）的数学原理和优势？

- **答案要点**：
  1. 核心思想：不对 embedding 直接加位置向量，而是对 Q/K 向量按位置 pos 施加一个二维旋转矩阵，使得 q_pos · k_pos' 的结果只依赖于相对距离 (pos - pos')。
  2. 数学形式：将 head_dim 维向量两两分组，每对视为复数，乘以旋转因子 e^(-iθ_i·pos)，其中 θ_i = 1 / 10000^(2i/d)。
  3. 优势：(a) 天然的相对位置编码性质；(b) 不需要额外参数；(c) 通过调整 base 频率可以外推到更长上下文（NTK-aware scaling、YaRN 等）。
- **追问点**：
  - RoPE 外推为什么难？怎么解决？（答：训练时见过的最大位置有限，推理时超出后旋转角度过大，Q·K 点积混乱；解决方法：位置插值（PI）把位置索引压缩、NTK-aware 调整 base 频率、YaRN 分段插值）
  - RoPE 对 GQA 有什么影响？（答：GQA 中 K/V 头少，RoPE 应用在每个 KV 头上，同组 Q 头共享同一组带位置编码的 K/V）

---

### Q10 【实战】如果让你优化一个长上下文（128K）Transformer 的推理性能，你会从哪些方面入手？

- **答案要点**：
  1. KV Cache 优化：启用 GQA 减少 KV 头数；KV Cache 量化到 INT8 甚至 INT4；PagedAttention 减少显存碎片。
  2. 注意力优化：滑动窗口注意力（Sliding Window Attention）只看最近 N 个 token；Ring Attention 跨设备切分序列；FlashAttention-2/3 提升 IO 效率。
  3. 系统层：Continuous Batching 提高 GPU 利用率；Chunked Prefill 混合 prefill 和 decode 请求；vLLM/SGLang 等推理框架。
  4. 模型层：检查模型是否支持长上下文外推（RoPE scaling）；必要时做长上下文继续预训练。
- **追问点**：
  - 128K 上下文的 KV Cache 显存大概多少？（答：7B 模型 GQA 8 KV 头，head_dim=128，28 层，FP16：2×28×8×128×131072×2bytes ≈ 15GB；batch 多条直接爆显存）
  - 滑动窗口注意力会不会丢信息？（答：会丢失远距离依赖，所以很多模型用全局+滑动窗口混合模式，如 Mistral 的每 4 层有 1 层全局注意力）

---

### Q11 【实战】为什么 Self-Attention 的复杂度是 O(n²·d)？在 n 很大时有什么替代方案？

- **答案要点**：
  1. 复杂度来源：QK^T 是 n×n 矩阵乘法（O(n²·d)），softmax 也是 n×n，再乘 V 又是 n²·d；序列长度 n 的平方项是主要瓶颈。
  2. 替代方案分类：
     - 稀疏注意力：只计算部分 token 对（如 Longformer 的滑动窗口+全局 token）。
     - 低秩近似：Linformer 把 K/V 投影到低维，将 n 压到 k（k<<n）。
     - 线性注意力：将 softmax 近似为核函数，分解为 φ(Q)·(φ(K)^T·V)，变为 O(n·d²)。
     - 状态空间模型：Mamba/SSM，完全回避注意力，O(n·d) 复杂度。
  3. 现状：精确注意力 + FlashAttention 在工程上已经够用（128K~1M），线性注意力和 SSM 仍在追赶精度。
- **追问点**：
  - Mamba 和 Transformer 比有什么优缺点？（答：优点是推理时 O(1) 状态、长序列极快；缺点是精确回忆能力弱于注意力、目前精度仍略逊于同规模 Transformer）

---
