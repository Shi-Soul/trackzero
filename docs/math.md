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

上训练网络，最简单的损失是力矩回归：

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

---

## 5. 覆盖率驱动的数据采集

proposal 的关键思想是：如果随机 rollout 覆盖不足，就用覆盖率目标主动扩展数据分布。

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

## 6. 基于已知动力学的模型驱动探索

proposal 的 Stage 1D 进一步利用“仿真器就是精确前向模型”这一优势。

### 6.1 可达集采样

从当前状态 $x$ 出发，$H$ 步可达集定义为

$$
\mathcal R_H(x)
=
\left\{
x_H \;\middle|\;
x_{t+1}=f_{\Delta t}(x_t,u_t),\ 
u_t\in\mathcal U,\ 
t=0,\dots,H-1
\right\}.
$$

与其从自然 rollout 分布采样，不如直接从 $\mathcal R_H(x)$ 中均匀或重加权采样目标状态：

$$
x_{t+H}^{\mathrm{goal}} \sim q(\cdot \mid \mathcal R_H(x_t)).
$$

这样得到的数据分布更接近“可控但少见”的区域。

### 6.2 轨迹优化生成训练数据

给定目标状态 $x_T^{\mathrm{goal}}$，可通过轨迹优化求解

$$
\min_{u_{0:T-1}}
\sum_{t=0}^{T-1}\|u_t\|_R^2
+ \|x_T - x_T^{\mathrm{goal}}\|_Q^2
$$

subject to

$$
x_{t+1}=f_{\Delta t}(x_t,u_t),\qquad u_t\in\mathcal U.
$$

所得最优轨迹

$$
\{(x_t^\star,u_t^\star)\}_{t=0}^{T-1}
$$

就可以作为覆盖导向的数据源。  
这和 Stage 1A 不同：这里的目标状态是为**覆盖空间**而选，不是来自某一种固定运动族。

### 6.3 规划蒸馏

如果在线规划器（如 iLQR / MPPI / CEM）在测试时能求出高质量动作

$$
u_t^{\mathrm{plan}} = \Pi(x_t, x_{t:T}^{\mathrm{ref}}),
$$

则可将其蒸馏为快速策略：

$$
\mathcal L_{\mathrm{distill}}(\theta)
=
\mathbb E
\left[
\|\pi_\theta(x_t,x_{t:T}^{\mathrm{ref}}) - u_t^{\mathrm{plan}}\|_2^2
\right].
$$

其目标是用离线学习换取在线推理速度。

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
\frac{1}{2}\|e_t^q\|_2^2,
$$

$$
\mathrm{MSE}_v
=
\frac{1}{T}\sum_{t=1}^{T}
\frac{1}{2}\|e_t^v\|_2^2.
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

## 8. 不可行参考的最优退化（Stage 2）

后续 proposal 还要研究**参考本身不可精确实现**的情况。记外部给定参考为 $\bar x_t^{\mathrm{ref}}$，它可能不满足真实动力学和力矩约束。

此时“最优行为”不再是精确跟踪，而是求一个受约束的最优折中：

$$
\min_{u_{0:T-1}}
\sum_{t=0}^{T-1}
\ell(x_t,\bar x_t^{\mathrm{ref}},u_t)
$$

subject to

$$
x_{t+1}=f_{\Delta t}(x_t,u_t),\qquad u_t\in\mathcal U.
$$

这里 $\ell$ 可以同时惩罚位置误差、速度误差和控制代价，例如

$$
\ell(x_t,\bar x_t^{\mathrm{ref}},u_t)
=
\|q_t-\bar q_t^{\mathrm{ref}}\|_{Q_q}^2
+ \|\dot q_t-\bar{\dot q}_t^{\mathrm{ref}}\|_{Q_v}^2
+ \|u_t\|_R^2.
$$

这给出了 proposal 中所谓 **optimal degradation baseline**：  
当参考不完全可达时，最优策略应当以物理一致的方式“尽量接近”，而不是盲目追逐不可实现目标。

---

## 9. 核心结论

TRACK-ZERO 的数学核心可以概括成四句话：

1. **动力学定义了可行性**：$x_{t+1}=f_{\Delta t}(x_t,u_t)$。
2. **逆动力学定义了跟踪本质**：找到从 $(x_t, x_{t+1}^{\mathrm{ref}})$ 到 $u_t$ 的映射。
3. **学习效果取决于覆盖率**：关键不是有没有标签，而是训练数据是否覆盖了足够广的可行转移空间。
4. **已知物理可替代动作先验**：通过随机探索、熵驱动覆盖、对抗生成和轨迹优化，可以仅凭物理与优化逼近通用跟踪策略。

因此，proposal 的主张可以写成一个简洁命题：

$$
\text{丰富且覆盖充分的可行物理数据}
\quad \Longrightarrow \quad
\text{可学习的通用逆动力学跟踪策略},
$$

而不必依赖任何人类动作先验分布。
