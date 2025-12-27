# LifelongAgentBench: Evaluating LLM Agents as Lifelong Learners

<p align="center">
    <img src="https://img.picui.cn/free/2025/05/21/682d857c0cb55.png" alt="Logo" width="80px">

[//]: # (    <br>)
[//]: # (    <b>WebArena is a standalone, self-hostable web environment for building autonomous agents</b>)
</p>

<p align="center">
<a href="https://www.python.org/downloads/release/python-3119/"><img src="https://img.shields.io/badge/python-3.11-blue.svg" alt="Python 3.11"></a>
<a href="https://pre-commit.com/"><img src="https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white" alt="pre-commit"></a>
<a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code style: black"></a>
<a href="https://mypy-lang.org/"><img src="https://img.shields.io/badge/mypy-strict-blue" alt="Checked with mypy"></a>
</p>

<p align="center">
<a href="https://caixd-220529.github.io/LifelongAgentBench/">ProjectPage</a> •
<a href="https://arxiv.org/abs/2505.11942">Paper</a> •
<a href="https://huggingface.co/datasets/csyq/LifelongAgentBench">Dataset</a>
</p>

# Setup

```shell
git clone ...
cd continual_agent_bench
pip install -r requirements.txt
pip install pre-commit==4.0.1  # ensure that pre-commit hooks are installed
pre-commit install  # install pre-commit hooks
pre-commit run --all-files  # check its effect

docker pull mysql  # build images for db_bench

docker pull ubuntu  # build images for os_interaction
docker build -f scripts/dockerfile/os_interaction/default scripts/dockerfile/os_interaction --tag local-os/default
```

# Run experiments
If you want to run experiments in single machine mode, please use the following command:
```shell
export PYTHONPATH=./
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python ./src/run_experiment.py --config_path "configs/assignments/experiments/llama_31_8b_instruct/instance/db_bench/instance/standard.yaml"
```

If you want to run experiments in distributed mode, you first need to start the `ServerSideController` in the machine that can deploy the docker containers.
```shell
export PYTHONPATH=./

python src/distributed_deployment_utils/server_side_controller/main.py
```
Then, you can run the following command in HPC node.
```shell
export PYTHONPATH=./
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python src/distributed_deployment_utils/run_experiment_remotely.py --config_path "configs/assignments/experiments/llama_31_8b_instruct/instance/db_bench/instance/standard.yaml"
```
The `ServerSideController` can be reused for multiple experiments.
> [!NOTE]
> Don't forget to update the IP address in `configs/components/environment.yaml` as well as in the files under `configs/components/clients`.

# Self-Evolving Agent (Experimental)

LifelongAgentBench now ships with a `SelfEvolvingController` that can autonomously craft lightweight Python tools and persist them for reuse.

- **Enable the controller:** switch the agent block to `self_evolving_agent` in `configs/assignments/experiments/llama_31_8b_instruct/agent.yaml` (see the commented example). The agent entry points to `src.self_evolving_agent.controller.SelfEvolvingController`.
- **Where tools live:** every generated tool is saved under the configured `tool_registry_path` (defaults to `outputs/{TIMESTAMP}/generated_tools/` in the agent component config). Metadata is persisted to `metadata.json` alongside the folder, and each tool is written as `<tool_name>.py` inside `generated_tools/`.
- **Inspecting logs:** add the optional `generated_tool_logging_callback` (commented in `configs/assignments/experiments/llama_31_8b_instruct/instance/db_bench/instance/standard.yaml`) to stream creation and invocation events into `outputs/<run>/generated_tools.log`.
- **Permissions:** ensure the process has write access to the configured `generated_tools/` directory; it is created automatically if missing.
- **Limitations:** actions must still be wrapped in `<action ...>` tags and task-specific output schemas (e.g., `Action: Operation` / `Action: Answer`) remain unchanged, so benchmark evaluation is identical to the baseline agent.
- **Forcing tool creation:** if you want to confirm the tool registry works without waiting for the model to emit `<action name="create_tool">`, add a `bootstrap_tools` block to `configs/components/agents/self_evolving_agent.yaml` (see the commented example) and the controller will pre-seed that tool on startup.
