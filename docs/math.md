# TRACK-ZERO 算法原理公式

本文把当前 research proposal 中的核心方法整理成一套自洽的数学描述，目标是说明：**只依赖物理、仿真和优化，而不依赖人类动作先验，也能学出通用的可行轨迹跟踪策略。**

---

## 1. 问题设定

以双摆为例，系统状态、控制量分别记为

$$
x_t = \begin{bmatrix} q_t \\ \dot q_t \end{bmatrix} \in \mathbb{R}^4,
\qquad
q_t \in \mathbb{R}^2,\ \dot q_t \in \mathbb{R}^2,
\qquad
u_t \in \mathbb{R}^2.
$$

控制受力矩约束：

$$
u_t \in \mathcal U
= \left\{u \in \mathbb{R}^2 : \|u\|_\infty \le \tau_{\max}\right\}.
$$

连续时间动力学写成标准机械系统形式：

$$
M(q)\ddot q + C(q,\dot q)\dot q + g(q) + D\dot q = u,
$$

其中 $M(q)$ 是质量矩阵，$C(q,\dot q)\dot q$ 是科氏/离心项，$g(q)$ 是重力项，$D\dot q$ 是阻尼项。  
经仿真器离散化后，一步转移可写为

$$
x_{t+1} = f_{\Delta t}(x_t, u_t).
$$

因此，TRACK-ZERO 的基本任务是：

> 给定当前状态 $x_t$ 和参考下一状态 $x_{t+1}^{\mathrm{ref}}$，输出一个力矩 $u_t$，使系统尽量跟踪参考轨迹。

---

## 2. 可行参考轨迹的建模

proposal 里首先只研究**可行参考**：参考轨迹本身就是由同一个动力学系统滚动得到的，因此原则上存在控制可精确跟踪。

### 2.1 多正弦激励

第 $i$ 个关节的参考力矩由多正弦信号生成：

$$
u_{t,i}^{\mathrm{ref}}
=
\operatorname{clip}
\left(
\sum_{k=1}^{K}
a_{i,k}\sin(2\pi f_{i,k} t\Delta t + \phi_{i,k}),
\ -\tau_{\max},\ \tau_{\max}
\right),
$$

其中：

$$
K \sim \mathcal{U}\{k_{\min},\ldots,k_{\max}\},
\qquad
f_{i,k} \sim \text{LogUniform}(f_{\min}, f_{\max}),
\qquad
a_{i,k} \sim \mathcal{U}\!\left(0,\frac{\tau_{\max}}{\sqrt K}\right),
\qquad
\phi_{i,k} \sim \mathcal{U}(0,2\pi).
$$

这种设计的含义是：

1. **平滑**：正弦叠加比白噪声更符合可控系统的实际输入。
2. **多尺度**：对数均匀采样频率，能同时覆盖慢变化和快变化模式。
3. **受限**：幅值和裁剪保证输入始终满足扭矩上界。

### 2.2 参考轨迹生成

给定初始状态 $x_0 \sim p_0(x)$，用参考力矩驱动系统得到

$$
x_{t+1}^{\mathrm{ref}} = f_{\Delta t}(x_t^{\mathrm{ref}}, u_t^{\mathrm{ref}}),
\qquad t=0,\dots,T-1.
$$

于是得到一条可行参考轨迹

$$
\tau^{\mathrm{ref}}
=
\left\{(x_t^{\mathrm{ref}}, u_t^{\mathrm{ref}})\right\}_{t=0}^{T-1}
\cup \{x_T^{\mathrm{ref}}\}.
$$

这类数据用于 Stage 0 / Stage 1A 的监督基线评估。

---

## 3. 逆动力学 oracle

如果动力学已知，那么从 $(x_t, x_{t+1}^{\mathrm{ref}})$ 直接反求力矩，本质上就是一步逆动力学问题。

### 3.1 有加速度时的精确逆动力学

若已知 $q_t,\dot q_t,\ddot q_t$，则理论所需力矩为

$$
u_t^\star
=
M(q_t)\ddot q_t + C(q_t,\dot q_t)\dot q_t + g(q_t) + D\dot q_t.
$$

这就是 MuJoCo `mj_inverse` 对应的连续时间逆动力学表达。

### 3.2 用下一时刻状态近似加速度

若只有 $x_t$ 和 $x_{t+1}^{\mathrm{ref}}$，可先用有限差分近似

$$
\ddot q_t \approx \frac{\dot q_{t+1}^{\mathrm{ref}} - \dot q_t}{\Delta t},
$$

再代回逆动力学方程，得到一个快速近似解：

$$
u_t^{\mathrm{fd}}
=
M(q_t)\frac{\dot q_{t+1}^{\mathrm{ref}} - \dot q_t}{\Delta t}
+ C(q_t,\dot q_t)\dot q_t + g(q_t) + D\dot q_t.
$$

### 3.3 Newton shooting 精确求解

为了获得与离散仿真器一致的“真”一步控制，proposal 里进一步把问题写成 shooting：

$$
u_t^\star
=
\arg\min_{u \in \mathcal U}
\left\|
\Pi_v\!\left(f_{\Delta t}(x_t,u)\right)
- \dot q_{t+1}^{\mathrm{ref}}
\right\|_2^2,
$$

其中 $\Pi_v(x)$ 表示从状态中取速度分量。

令残差为

$$
r(u)
=
\Pi_v\!\left(f_{\Delta t}(x_t,u)\right)
- \dot q_{t+1}^{\mathrm{ref}},
$$

则 Newton 迭代写成

$$
u^{(k+1)} = u^{(k)} - J(u^{(k)})^\dagger r(u^{(k)}),
$$

其中

$$
J(u) = \frac{\partial r(u)}{\partial u},
$$

$J^\dagger$ 是伪逆或最小二乘解。实现上，$J$ 可用有限差分近似。  
这个 oracle 给出了 proposal 中的**性能上界**。

---

## 4. 学习型逆动力学策略

学习策略用神经网络逼近一步逆映射：

$$
\hat u_t = \pi_\theta(x_t, x_{t+1}^{\mathrm{ref}}).
$$

若使用多步前视窗口，也可写成

$$
\hat u_t = \pi_\theta\!\left(x_t, x_{t+1:t+H}^{\mathrm{ref}}\right),
$$

其中 $x_{t+1:t+H}^{\mathrm{ref}}$ 表示未来 $H$ 步参考状态。

### 4.1 监督基线（Stage 1A）

在参考数据集

$$
\mathcal D_{\mathrm{sup}}
=
\left\{
(x_t^{(n)}, x_{t+1}^{(n)}, u_t^{(n)})
\right\}
$$

上训练网络，损失是**原始力矩**回归（注意：不做力矩归一化）：

$$
\mathcal L_{\mathrm{sup}}(\theta)
=
\mathbb E_{(x_t,x_{t+1},u_t)\sim \mathcal D_{\mathrm{sup}}}
\left[
\|\pi_\theta(x_t,x_{t+1}) - u_t\|_2^2
\right].
$$

它学习的是“参考数据分布上的”逆动力学，因此分布内表现强，但未必能外推到新的可行运动族。

### 4.2 TRACK-ZERO 自监督（Stage 1B）

不使用目标参考数据，而是自己采样控制并滚动系统：

$$
u_t \sim p_\eta(u_t \mid u_{<t}, x_{\le t}),
\qquad
x_{t+1}=f_{\Delta t}(x_t,u_t).
$$

从而构造自生成数据集

$$
\mathcal D_{\mathrm{self}}
=
\left\{
(x_t, x_{t+1}, u_t)
\right\}.
$$

训练目标与监督基线相同：

$$
\mathcal L_{\mathrm{self}}(\theta)
=
\mathbb E_{(x_t,x_{t+1},u_t)\sim \mathcal D_{\mathrm{self}}}
\left[
\|\pi_\theta(x_t,x_{t+1}) - u_t\|_2^2
\right].
$$

核心问题不在损失函数，而在于 **$\mathcal D_{\mathrm{self}}$ 是否覆盖了足够广的可行状态-转移空间**。

### 4.3 训练优化发现（Stage 1D）

Stage 1D 的核心发现：**探索策略对性能影响有限，训练优化才是关键**。使用 Cosine LR + Weight Decay（WD=1e-4）在 10K 数据上实现 47% 性能提升，而 bangbang 控制、覆盖率采样等数据工程方案没有统一改善效果。

推荐配置：AdamW, lr=1e-3, WD=1e-4, CosineAnnealingLR。
模型容量排序：$1024\times6 > 512\times4 > 256\times3$（10K 数据下不过拟合）。
数据规模相变在 50K（10K→50K 约 4× 提升，50K→100K 趋于饱和）。

### 4.4 基于物理结构的架构（Stage 2A / Stage 3C）

**残差 PD 架构**（Stage 2A）将输出分解为反馈项和前馈项：

$$
\tau = K_p(x)\odot\Delta q + K_d(x)\odot\Delta v + \tau_{\mathrm{ff}}(x),
$$

其中 $K_p,K_d>0$（softplus 保证），网络输出 $3n_q$ 维。在 2 DOF 上比原始 MLP 好 9.7×，但高 DOF 下优势衰减（5 DOF 仅 1.6×）。

**分解动力学架构**（Stage 3C）利用逆动力学在加速度上的线性结构：

$$
\tau = A(q,\dot q)\cdot[\Delta q;\,\Delta v] + b(q,\dot q),
$$

$A\in\mathbb{R}^{n_q\times 2n_q}$ 是全增益矩阵（类比 $M(q)$），$b$ 是状态相关偏置（类比重力+科氏项）。两个子网络只接受当前状态 $x_t$，跟踪目标通过线性项引入，与物理结构完全一致。

DOF 扩展结果（512×4，2K 轨迹，200 轮）：

| DOF | 原始 MLP | 残差 PD | 分解架构 | vs MLP |
|-----|---------|---------|---------|--------|
| 2   | 3.78e-4 | 3.87e-4 | **9.22e-5** | 4.1× |
| 3   | 2.55e-1 | 9.87e-2 | **3.33e-3** | 76× |
| 5   | 1.07    | 2.69e-1 | **1.87e-2** | 57× |

分解架构的优势随 DOF **增大**，残差 PD 的优势随 DOF **衰减**——正好相反。

---

## 5. 覆盖率驱动的数据采集（理论方向，待深入验证）

Stage 1C/1D 实验表明探索策略对 benchmark 改善有限；但覆盖率驱动的数据采集仍是长远的重要方向，以下为理论描述。

### 5.1 状态分布熵最大化

记采样分布诱导的状态边缘分布为 $p_\eta(x)$。理想目标是增大访问分布熵：

$$
\max_\eta H\!\left(p_\eta\right)
=
\max_\eta\left(
- \int p_\eta(x)\log p_\eta(x)\,dx
\right).
$$

离散 bin 版本可写为

$$
H_{\mathrm{bin}}
=
-\sum_{b=1}^{B}\hat p_b \log \hat p_b,
$$

其中 $\hat p_b$ 是第 $b$ 个状态 bin 的占据频率。  
这对应 proposal 中的 **state-space binning with rebalancing**。

### 5.2 稀有状态奖励

如果用 RL 训练一个“数据采集策略” $\eta$，可定义覆盖奖励为

$$
r_t^{\mathrm{cov}} = -\log \hat p(x_t),
$$

或在离散 bin 下写成

$$
r_t^{\mathrm{cov}} = \frac{1}{\sqrt{N(b(x_t)) + 1}},
$$

其中 $N(b)$ 是对应 bin 的访问计数。  
这会鼓励策略访问低密度区域。

### 5.3 集成分歧（ensemble disagreement）

训练多个逆动力学模型 $\{\pi_{\theta_m}\}_{m=1}^M$，在某个转移上的不确定性可表示为

$$
\mu(x_t,x_{t+1}) = \frac1M\sum_{m=1}^M \pi_{\theta_m}(x_t,x_{t+1}),
$$

$$
d(x_t,x_{t+1})
=
\frac1M \sum_{m=1}^M
\left\|
\pi_{\theta_m}(x_t,x_{t+1}) - \mu(x_t,x_{t+1})
\right\|_2^2.
$$

然后优先采集高分歧样本：

$$
\max_\eta \mathbb E_{\tau \sim p_\eta}
\left[
\sum_{t=0}^{T-1} d(x_t,x_{t+1})
\right].
$$

这比单纯看状态密度更直接，因为它瞄准的是**逆动力学模型尚不确定的区域**。

### 5.4 Hindsight relabeling

如果当前策略跟踪某参考失败，但实际产生了一条可行轨迹 $\tau^{\mathrm{act}}=\{x_t^{\mathrm{act}},u_t\}$，则可把它重标记为新的成功参考：

$$
\tilde x_t^{\mathrm{ref}} \leftarrow x_t^{\mathrm{act}}.
$$

于是得到额外训练样本

$$
(\,x_t^{\mathrm{act}},\ \tilde x_{t+1}^{\mathrm{ref}},\ u_t\,),
$$

本质上是“把失败 rollout 转化为可监督的可行数据”。

### 5.5 对抗式参考生成

再进一步，可训练一个参考生成器 $g_\phi$ 专门寻找最难跟踪的可行轨迹：

$$
\tau_\phi = g_\phi(z), \qquad \tau_\phi \text{ 满足真实动力学约束}.
$$

形成如下 min-max 问题：

$$
\min_\theta \max_\phi\ 
\mathbb E_{z}
\left[
\mathcal E\bigl(\pi_\theta,\tau_\phi\bigr)
\right],
$$

其中 $\mathcal E$ 是闭环跟踪误差。  
这个机制会把采样集中到**可行集边界附近**和当前策略的薄弱区域。

---

## 6. 训练优化与数据规模（Stage 1D 实验发现）

Stage 1D 的核心贡献是确定了两个设计轴的实际效果。

### 6.1 训练优化：Cosine LR + Weight Decay

固定数据规模（10K 轨迹），Cosine LR + WD=1e-4 相比 Fixed LR 无 WD 的基线
获得约 **47% 的 benchmark 性能提升**。数据工程方案（bangbang、覆盖率采样）均未能复现类似改善。

推荐配置（已验证）：
- 优化器：AdamW，lr=1e-3，weight_decay=1e-4
- 调度器：CosineAnnealingLR(T_max=epochs)

### 6.2 数据规模的相变

数据量对性能的影响呈非线性，相变点在 50K 轨迹附近：

| 规模变化 | 效果 |
|---------|------|
| 10K → 20K | +1%（近乎无效） |
| 10K → 50K | 约 4× 提升 |
| 50K → 100K | 趋于饱和 |

### 6.3 模型容量

当前数据规模下不存在过拟合，更大网络持续更好：

$$
1024\times6 > 512\times4 > 256\times3.
$$

高自由度（3 DOF 以上）下，10× 额外数据无提升，根本瓶颈是架构表达能力，
需通过分解动力学架构（§4.4）解决，而非堆数据。
---

## 7. 闭环跟踪与评估指标

无论策略如何训练，最终都在闭环里评估。给定参考轨迹 $\{x_t^{\mathrm{ref}}\}$，系统按

$$
x_{t+1}
=
f_{\Delta t}\!\left(
x_t,\ \pi_\theta(x_t, x_{t+1}^{\mathrm{ref}})
\right)
$$

滚动。

### 7.1 角度误差

关节角是周期变量，因此误差要取 wrap 形式：

$$
e_t^q
=
\operatorname{wrap}(q_t^{\mathrm{ref}} - q_t)
=
\operatorname{atan2}\!\bigl(
\sin(q_t^{\mathrm{ref}}-q_t),
\cos(q_t^{\mathrm{ref}}-q_t)
\bigr).
$$

速度误差为

$$
e_t^v = \dot q_t^{\mathrm{ref}} - \dot q_t.
$$

### 7.2 轨迹级指标

对一条长度为 $T$ 的轨迹，定义

$$
\mathrm{MSE}_q
=
\frac{1}{T}\sum_{t=1}^{T}
\frac{1}{n_q}\|e_t^q\|_2^2,
$$

$$
\mathrm{MSE}_v
=
\frac{1}{T}\sum_{t=1}^{T}
\frac{1}{n_q}\|e_t^v\|_2^2.
$$

仓库当前评估脚本使用的总误差是

$$
\mathrm{MSE}_{\mathrm{total}}
=
\mathrm{MSE}_q + \lambda_v \mathrm{MSE}_v,
\qquad
\lambda_v = 0.1.
$$

此外还统计

$$
\max_t \|e_t^q\|_\infty,
\qquad
\operatorname{percentile}_{95}\!\left(\{|e_t^q|\}_{t=1}^T\right),
$$

用于识别偶发的大误差尾部。

---

## 8. 噪声参考与鲁棒性（Stage 2C）

Stage 2C 研究了**参考轨迹被噪声污染**的情况（位置加 Gaussian 噪声 $\sigma$，速度加 $5\sigma$）。MLP 的隐式鲁棒性：MLP 与 oracle 的优劣有一个交叉点 $\sigma\approx0.005$。
低噪声时 oracle 精确计算占优；高噪声时 MLP 的光滑函数近似起到低通滤波效果：

| $\sigma$ | MLP AGG | Oracle AGG | MLP/Oracle |
|-----------|---------|------------|------------|
| 0.00 | 6.80e-4 | 7.63e-5 | 8.9× 差 |
| 0.01 | 1.37e-2 | 6.99e-2 | **5.1× 好** |
| 0.05 | 2.66e-1 | 6.40e-1 | 2.4× 好 |
| 0.10 | 5.76e-1 | 1.08 | 1.9× 好 |

**噪声增广训练**（训练时 50% 样本加 $\sigma=0.05$ 噪声）形成精度-鲁棒性权衡：

$$
\text{clean 性能下降 78×},\qquad \sigma=0.05 \text{ 下性能提升 4.5×}.
$$

这说明 MLP 对噪声的隐式鲁棒性是一个内在优势，可通过训练显式增强，
但代价是精确跟踪性能的损失——应根据实际应用场景选择。

---

## 9. 核心结论

TRACK-ZERO 目前的实验结论可以概括为以下几点：

1. **动力学定义了可行性**：$x_{t+1}=f_{\Delta t}(x_t,u_t)$，逆映射 $(x_t,x_{t+1})\to u_t$ 是跟踪的本质。
2. **架构 >> 数据量**：残差 PD（Stage 2A）、分解动力学（Stage 3C）每次都带来量级改善，
   而 5× 更多数据通常只带来 <2× 提升甚至无提升。
3. **训练优化 >> 探索策略**：Cosine LR + WD 在 10K 数据上带来 47% 提升，
   bangbang/覆盖率采样等探索方案无统一效果。
4. **线性加速度结构是关键感应偏置**：分解架构之所以随 DOF 改善更多，
   正是因为它准确编码了逆动力学在加速度上的线性性。
5. **开环验证误差 ≠ 闭环跟踪性能**：误差坐标输入和多步参考上下文在 val loss 上更好，
   但在闭环 benchmark 上更差，因为它们放大了闭环分布偏移。

因此 TRACK-ZERO 的核心主张更新为：

$$
\text{随机可行数据} + \text{物理结构化架构} + \text{Cosine+WD 训练}
\;\Longrightarrow\;
\text{通用逆动力学跟踪策略},
$$

不依赖人类动作先验，且优势随 DOF 增大而增大（分解架构）。
