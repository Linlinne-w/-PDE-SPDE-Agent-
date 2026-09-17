# PDE/SPDE Research Agent

目标：把“求解器调用”升级为一个**可运行、可评分、可纠错**的科研 Agent 工作流。适用任务：对给定 PDE/SPDE，在计算预算内自动选择时间步长与空间分辨率，满足误差要求并产出可复现实验结论。

## 1. Agent 闭环

Agent 执行以下闭环：

1. 理解方程与约束（边界/初值、噪声项、预算、误差阈值）
2. 选择数值方法（时间离散、空间离散、稳定性策略）
3. 调用求解器并记录配置
4. 验证结果（误差、收敛阶、稳定性、预算占用）
5. 诊断失败原因（发散、超时、内存不足、精度不达标）
6. 自动调整参数后重算
7. 输出代码版本、实验数据、图表与结论

## 2. 项目结构与运行产物

当前仓库实现了一维确定性热方程和具有显式解的随机热方程：

```text
.
├── solver/                 # PDE/SPDE 求解器实现
├── tasks/                  # 任务配置与字段校验
├── evaluator/              # 评分与验收逻辑
├── tests/                  # 确定性与随机求解器测试
├── run_agent.py            # 端到端执行与自动重试
├── runs/                   # 运行时动态产物（不提交）
└── reports/                # 自动生成的实验报告
```

每次运行都会生成：

- `runs/<run_id>/config.json`：完整参数与随机 seed
- `runs/<run_id>/metrics.json`：误差、耗时、内存、评分和尝试记录
- `runs/<run_id>/artifacts/error_history.csv`：各轮参数与误差
- `runs/<run_id>/artifacts/solution_final.csv`：最终网格、数值解、参考解和绝对误差
- `runs/<run_id>/artifacts/solution_comparison.png`：解与误差对比图（需要 `matplotlib`）
- `reports/<run_id>.md`：自动结论与诊断建议

## 3. 任务输入与输出契约

任务使用 JSON 描述方程、区域、初值/边界条件、误差阈值、资源预算、seed 和最大重试次数。当前支持：

- `equation: "heat_1d"`
- `equation: "stochastic_heat_1d"`

`initial_condition` 和 `boundary_condition` 当前是描述字段，不解析任意表达式。未知方程会被拒绝；随机热方程必须提供 `sigma`。

## 4. 评分体系

总分 100：正确性 50、效率 20、鲁棒性 20、可复现性 10。硬门槛为误差达标、稳定、未超时、未超内存；失败时会记录失败类型和下一轮修正建议。

## 5. 自动纠错机制

失败后按失败类型调整参数：精度不足或发散时减小 `dt`、加密网格；超时或内存超限时适当放宽网格并增大步长，但始终限制在扩散 CFL 稳定范围内。每轮尝试都会保留，直到通过或耗尽最大重试次数。

## 6. 确定性热方程示例

```bash
python run_agent.py --task tasks/heat_equation_1d.json
```

核心实现仅依赖 Python 标准库；安装 `matplotlib` 后会额外生成对比图。确定性示例使用解析解 `u(x,t)=exp(-alpha*pi^2*t)sin(pi*x)` 进行误差验证。

## 7. 新增 SPDE：具有显式解的随机热方程

当前支持的随机问题为 **Itô 型、标量时间乘性噪声**：

$$
du(t,x)=\alpha u_{xx}(t,x)\,dt+\sigma u(t,x)\,dB_t,
\qquad x\in(a,b).
$$

边界为 $u(t,a)=u(t,b)=0$，初值为
$u(0,x)=\sin(\pi(x-a)/(b-a))$。其中 $B_t$ 为一维标准布朗运动，所有空间点共享这一时间噪声；该算例不使用空间白噪声或一般的 Q-Wiener 噪声。

令 $L=b-a$，显式解为

$$
u(t,x)=\exp\left[\left(-\alpha\frac{\pi^2}{L^2}-\frac{\sigma^2}{2}\right)t+\sigma B_t\right]
\sin\left(\frac{\pi(x-a)}{L}\right).
$$

数值方法为**空间二阶中心差分 + 时间 Euler–Maruyama**：

$$
U_i^{n+1}=U_i^n+\frac{\alpha h_n}{\Delta x^2}
(U_{i+1}^n-2U_i^n+U_{i-1}^n)+\sigma U_i^n\Delta B_n.
$$

最后一步使用实际剩余时间 $h_n$。数值解和显式解使用同一路径。参数重试复用 `BrownianPath`，通过条件布朗桥补充新时间点并保留已采样值；仅为不同时间网格设置相同 seed 并不足以耦合路径。

运行随机任务：

```bash
python -m pip install matplotlib  # 可选，用于绘图
python run_agent.py --task tasks/stochastic_heat_equation_1d.json
python -m unittest discover -s tests -v
```

输出额外包括：

- `runs/<run_id>/artifacts/brownian_path.csv`：最终一次求解的时间点、布朗值和实际增量
- `runs/<run_id>/artifacts/solution_final.csv`：同一路径上的数值解、显式解和绝对误差
- `runs/<run_id>/artifacts/solution_comparison.png`：同一路径上的解对比图

`solve_stochastic_heat_equation_1d(config, brownian_increments=...)` 可以重放保存的增量。随机任务的 `l2_error` 是**单条轨道的空间 L2 误差**，不是多轨道强误差的蒙特卡洛估计；`estimated_convergence_order` 为 `null`。稳定性门槛检查确定性扩散 CFL 条件与输出有限性，不等价于随机格式的均方稳定性证明。

## 8. 验收清单

- [ ] 同一任务可一键复现（含依赖、命令、seed）
- [ ] 输出包含误差表和收敛图
- [ ] 评分脚本可自动给分并输出明细
- [ ] 失败时有可解释诊断与自动重试记录
- [ ] 报告包含方法、参数、结果、结论和局限

## 9. 项目定位

这个仓库面向腾讯北京混元基座 STEM Agent 研究实习方向，目标不是“单次算对”，而是构建一个能持续迭代的科研 Agent：任务可标准化输入，结果可自动验收，失败可结构化纠错，全流程可复现、可比较、可追踪。项目覆盖可执行科研工作流、科学计算工具调用、自动验证、Reward/评测指标和长程 Agent 的典型失败模式分析。
