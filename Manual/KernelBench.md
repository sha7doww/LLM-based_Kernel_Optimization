## 文件介绍



## 使用方法

### Run on a single problem

生成 Kernel 并进行正确性测试、测量编译时间和运行时间（不会计算基线和加速比）：

```bash
python3 scripts/generate_and_eval_single_sample.py         
dataset_src="huggingface"  \
level=1  \
problem_id=101  \         
server_type=openai  \
model_name="o3"  \
gpu_arch="['Ampere']"  \
log=true
```

运行上一步得到的 Kernel 和基线并进行比较：

```bash
python3 scripts/run_and_check.py  \
ref_origin=kernelbench  \
level=2 problem_id=40  \
kernel_src_path=results/eval_logs/generated_kernel_level_2_problem_40.py  \
gpu_arch="['Ampere']"
```


### Run on all problems

```bash
# 1. Generate responses and store kernels locally to runs/{run_name} directory
python3 scripts/generate_samples.py  \
run_name=test_hf_level_1  \
dataset_src=huggingface  \
level=1  \
num_workers=50  \
server_type=deepseek  \
model_name=deepseek-chat  \
temperature=0
```

```bash
# 2. Evaluate on all generated kernels in runs/{run_name} directory
python3 scripts/eval_from_generations.py  \
run_name=test_hf_level_1  \
dataset_src=local  \
level=1  \
num_gpu_devices=8  \
timeout=300
```

```bash
# 3. Generate the baseline time
generate_baseline_time.py
```

```bash
# 4. Analyze the eval results to compute Benchmark Performance
python3 scripts/benchmark_eval_analysis.py  \
run_name=test_hf_level_1  \
level=1  \
hardware=L40S_matx3  \
baseline=baseline_time_torch
```
