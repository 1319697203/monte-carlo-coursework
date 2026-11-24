# 软件项目工期风险评估

采用 Beta-PERT 工期采样与动态关键路径计算，分析校园报修系统的项目工期风险。输入为教学假设数据。

## 环境与安装
Python 3.12，依赖版本见 requirements.txt。在本目录执行：
```bash
python -m pip install -r requirements.txt
```

## 运行仿真与绘图
```bash
python -X utf8 src/simulate.py --n 50000 --seed 20261024 --deadline 35 --out results_reproduced
```
程序生成基准、风险缓解、悲观尾部扩张三个情景，各50000次。结果写入 results_reproduced；随附 results 目录提供论文对应结果。基准种子为20261024，另两个情景分别使用20261025、20261026。

## 模型测试
```bash
python -X utf8 -m unittest discover -s tests -v
```
测试涵盖并行汇合、关键分支切换、多终点、等长关键路径、采样均值和退化工期。

## 文件说明
- data/tasks.csv：任务、依赖、三点工期和估计理由。
- src/simulate.py：仿真、统计分析和绘图。
- tests/test_model.py：模型测试。
- results/summary.json：实验配置与汇总指标。
- results/*.csv：排程、敏感度、截止概率与稳定性统计。
- results/simulation_samples.npz：模拟样本与关键任务标记。
- results/figures/：六组PNG/SVG图。

绘图使用英文标签与任务ID，中文含义见论文WBS表。论文及其LaTeX工程另附，使用XeLaTeX编译。模型假设任务工期独立，未考虑共享人力约束；风险缓解情景不代表已验证的现实效果。
