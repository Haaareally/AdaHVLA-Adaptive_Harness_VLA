<div align="center">

# AdaHVLA

### Adaptive Harnesses for Long-Horizon Vision-Language-Action Execution

**Adaptive coordination for long-horizon robot tasks.**

The preprint of our paper is coming soon.

[Overview](#overview) · [Architecture](#architecture) · [Customizing the harness](#customizing-the-harness) · [Getting started](#getting-started) · [Running adaptation](#running-adaptation) · [Acknowledgments](#acknowledgments)

</div>

AdaHVLA connects task-level reasoning and memory with vision-language-action (VLA) execution through an adaptive harness. Long-horizon tasks require a robot to remember what happened, track unfinished goals, and respond when execution departs from the plan. A locally appropriate action alone may not establish the conditions needed for the next stage: reaching a doorway, for example, is different from passing through it.

Our central idea is to make these coordination decisions explicit and editable in code. The harness determines what instruction and visual history the VLA receives, when task progress is committed, and how recovery and completion are handled. Between rollouts, a multi-agent process turns robot experience into testable hypotheses and code revisions, then checks their effects in subsequent execution. A revision graph retains alternative harnesses and the evidence behind them, allowing successful and unsuccessful attempts to inform continued adaptation across tasks and environments.

<p align="center">
  <img src="docs/assets/figure1.png" width="560" alt="Paper Figure 1: AdaHVLA in simulated navigation and manipulation environments and on a real quadruped." />
</p>
<p align="center"><em>Figure 1. Deployments across tasks, environments, and robotic platforms studied in the paper.</em></p>

## Overview

This release provides the **Go2 quadruped navigation** implementation: the online harness, multi-agent adaptation loop, NaVILA service adapter, simulator configuration, and a 50-episode benchmark packaged as `navila-LH`.

- **Persistent task context.** Recent observations, visual checkpoints, and compact memory support reasoning across execution stages.
- **Editable coordination policies.** Python policies govern subgoal progress, local instructions, visual refresh, recovery, and completion.
- **Evidence-driven adaptation.** Separate manager, analyst, engineer, and reviewer contexts connect observed failures to focused revisions and behavioral comparisons.
- **Cumulative adaptation memory.** A revision graph links evidence, hypotheses, candidate code, and observed effects to guide later attempts.

The navigation implementation uses [NaVILA](https://github.com/AnjieCheng/NaVILA) to produce navigation actions and a pretrained Go2 locomotion controller to execute them.

The paper explores adaptation across robot platforms. This code release currently includes the Go2 simulation path; manipulation and real-robot adapters are not included. For manipulation research, we strongly recommend [π0.5](https://www.pi.website/blog/pi05) and the [openpi implementation](https://github.com/Physical-Intelligence/openpi).

## Architecture

![Paper Figure 2: execution harness, evidence-driven adaptation loop, and candidate revision graph.](docs/assets/figure2.png)

*Figure 2. During execution, coordination policies connect task reasoning and visual history to the VLA. During adaptation, rollout evidence informs hypotheses, source revisions, and comparisons recorded in a revision graph.*

### Online execution

```text
Task + visual observations
          ↓
Harness: context → subgoal progress → instruction and context handoff
          ↓
NaVILA → navigation action → Go2 locomotion controller
          ↓
Isaac Sim → updated observations → next harness decision
```

The harness combines recent observations, visual checkpoints, a compact task memory, and execution feedback. A single reasoning interface proposes a plan and subsequent progress decisions. Explicit policies validate those decisions and control when to refresh the VLA's local history. Task-level memory survives a local history refresh. The VLA owns motion generation; the harness supplies objectives and can declare task completion.

### Adaptation between rollouts

```text
Prototype rollout → analyze evidence → propose a harness revision
       ↑                                      ↓
Select / continue ← compare parent and child ← check, review, and evaluate
       ↓
Selected harness → held-out evaluation
```

The starting harness is saved as `C0000`. The engineer can edit `harness.py` in a candidate copy. Each proposed revision states a coordination hypothesis and its expected observable effect. Source checks and independent review determine whether a candidate can run; new rollouts test whether the mechanism actually changed behavior. Parent–child comparisons guide selection, while the revision graph retains the evidence for further adaptation. Candidate copies and hashes provide integrity checks, not an operating-system security sandbox.

| File | Responsibility |
| --- | --- |
| [`harness.py`](src/adahvla/harness.py) | Reasoning prompt, visual context, progress and handoff policies, online loop |
| [`adaptation.py`](src/adahvla/adaptation.py) | Agent contexts, revision workflow, selection, budgets, and held-out phase |
| [`workspace.py`](src/adahvla/workspace.py) | Candidate source, checks, subprocess evaluation, and evidence storage |
| [`vla.py`](src/adahvla/vla.py) | NaVILA socket client and inference server |
| [`locomotion.py`](src/adahvla/locomotion.py) | Controller loading, observations, and velocity-command execution |
| [`configs/`](configs/) | Go2, sensors, simulation timing, and asset paths |
| [`scripts/run.sh`](scripts/run.sh) | Prototype → adaptation → held-out evaluation entry point |
| [`scripts/evaluate.py`](scripts/evaluate.py) | One candidate and one episode in a separate simulator process |
| [`tests/`](tests/) | Offline software tests |

## Customizing the harness

**The included harness is a sample implementation and a starting point for your own robot tasks.** Its prompts, memory structure, progress rules, and recovery behavior illustrate the AdaHVLA approach. For a specific robot, task family, or environment, you should write or optimize these policies around the robot's capabilities, available observations, and observable task-completion criteria.

Start with [`harness.py`](src/adahvla/harness.py): adapt `DECISION_PROMPT` to the task, `ContextPolicy` to the useful observation history, and `ProgressPolicy` and `HandoffPolicy` to stage transitions and execution guidance. When integrating another robot or VLA, also implement the corresponding executor and environment interfaces and define an appropriate evaluation metric. The supplied adapters and runner currently target Go2 navigation.

Validate your starting harness on representative rollouts, then use the adaptation loop to investigate and refine its coordination behavior. Check each revision against its predicted effect and evaluate the selected harness on tasks reserved for testing.

## Getting started

Run the following commands from the **AdaHVLA project root**. Core harness and adaptation code use Python 3.10+ and the standard library. Full navigation runs additionally require the simulator, a NaVILA server, and a vision-capable reasoning API.

### 1. Check the source and assets

For an offline check, no model service or GPU is needed:

```bash
cd /path/to/AdaHVLA
python -m pip install -r requirements.txt
bash scripts/run.sh --check
```

This verifies asset hashes, episode selections, and the prototype interface without calling an API, starting Isaac Sim, or creating a persistent experiment.

Expected resources:

```text
benchmarks/navila-LH/dataset.json.gz   # 50 navigation episodes
assets/locomotion/go2/policy.jit       # Go2 inference controller
assets/robots/go2/                    # robot USD and meshes
assets/matterport/<scene>/            # six scenes and their textures
assets/manifest.json                  # asset provenance, sizes, and SHA-256 hashes
```

`navila-LH` is the benchmark name used in this release. The environment and asset setup builds on [NaVILA-Bench](https://github.com/yang-zj1026/NaVILA-Bench) and its [VLN-CE-Isaac data release](https://huggingface.co/datasets/Zhaojing/VLN-CE-Isaac). The run selections below are configurable examples, not an upstream benchmark protocol. Keep scene textures with their USD files when copying the project.

### 2. Install the simulation environment

The simulation baseline follows [NaVILA-Bench's installation requirements](https://github.com/yang-zj1026/NaVILA-Bench#installation): **Linux, Python 3.10, Isaac Sim 4.1.0, and the NaVILA Isaac Lab 1.1.0 fork**, with a compatible NVIDIA GPU and driver. For the pip-based installation below, use Ubuntu 22.04 or later.

```bash
conda create -n adahvla-isaac python=3.10 -y
conda activate adahvla-isaac

cd /path/to/AdaHVLA
git clone https://github.com/yang-zj1026/IsaacLab.git /path/to/IsaacLab
ISAACLAB_ROOT=/path/to/IsaacLab bash scripts/setup_locomotion.sh
```

The helper validates the Lab checkout, installs Lab core and [`requirements-locomotion.txt`](requirements-locomotion.txt), and runs `pip check`. It selects PyTorch 2.2.2 with CUDA 12.1 wheels by default; set `ADAHVLA_CUDA=cu118` to use CUDA 11.8 wheels. Additional VLNCE/Matterport extensions and RSL-RL are not needed for this inference path.

The controller consumes 909 values: 45 proprioceptive features, a 459-value height map, and nine frames of proprioceptive history. It outputs 12 joint actions at 50 Hz. Timing and resource paths are defined in [`configs/locomotion.json`](configs/locomotion.json); relative paths resolve from the project root.

### 3. Start the NaVILA server

Install the model dependencies in a **separate environment** using the [NaVILA repository](https://github.com/AnjieCheng/NaVILA) and [Isaac evaluation setup](https://github.com/yang-zj1026/NaVILA-Bench#vla-evaluation). Download the [NaVILA Llama-3 8B, 8-frame checkpoint](https://huggingface.co/a8cheng/navila-llama3-8b-8f). VLA weights are not bundled here.

In **terminal A**, activate that environment and launch this project's server adapter:

```bash
conda activate navila
PYTHONPATH=/path/to/AdaHVLA/src${PYTHONPATH:+:$PYTHONPATH} \
  python -m adahvla.vla \
  --model-path /path/to/navila-llama3-8b-8f \
  --host 127.0.0.1 --port 54321 --device cuda
```

Wait for `NaVILA listening` and leave the server running. The server uses an unauthenticated socket protocol; keep it on loopback or access it through a trusted private connection, such as an SSH tunnel.

### 4. Configure the reasoning API

In **terminal B**, activate the simulation environment:

```bash
conda activate adahvla-isaac
cd /path/to/AdaHVLA

export ADAHVLA_BASE_URL="https://your-api-endpoint/v1"
export ADAHVLA_MODEL="your-vision-model"
read -rs -p "API key: " ADAHVLA_API_KEY; echo
export ADAHVLA_API_KEY
```

The endpoint must support the Chat Completions format, image inputs, JSON object responses, and `temperature=0`. The online harness and four adaptation roles use the same model configuration with separate contexts.

[`scripts/run.sh`](scripts/run.sh) also contains `YOUR_API_KEY`, `YOUR_API_ENDPOINT`, and `YOUR_VISION_MODEL` placeholders that you can fill locally. Environment variables take precedence. Prefer the terminal setup above when sharing code, and never commit a script containing a real key. The API key is passed through the environment rather than command-line arguments.

| Variable | Purpose | Default |
| --- | --- | --- |
| `ADAHVLA_API_KEY` | Reasoning API credential | User supplied |
| `ADAHVLA_BASE_URL` | API base URL, usually ending in `/v1` | User supplied |
| `ADAHVLA_MODEL` | Vision-capable model with JSON output | User supplied |
| `ADAHVLA_VLA_HOST` | NaVILA server address | `127.0.0.1` |
| `ADAHVLA_VLA_PORT` | NaVILA server port | `54321` |
| `ADAHVLA_PYTHON` | Python executable for the run script | `python` |

## Running adaptation

Start with one prototype rollout to verify the complete model–simulator connection:

```bash
bash scripts/run.sh --prototype-only
```

Then continue the same session through adaptation and held-out evaluation:

```bash
bash scripts/run.sh
```

You can also run the second command directly: it evaluates the prototype first.

The defaults use **episode 27 for adaptation, 124 for validation, and 167 for held-out testing**. These three episodes provide a small starting configuration. They do not reproduce the paper's full evaluation. All selection arguments refer to **`episode_id`**, not dataset row numbers or `episode_new_id`.

To choose disjoint episode sets and an explicit run directory:

```bash
bash scripts/run.sh \
  --train-ids 27,124 \
  --validation-ids 167 \
  --test-ids 191 \
  --max-steps 16 --max-candidates 4 --max-rollouts 12 \
  --output-dir ../AdaHVLA-runs/experiment-01
```

| Option | Meaning | Default |
| --- | --- | --- |
| `--max-steps` | Manager scheduling steps | `16` |
| `--max-candidates` | Revision attempts | `4` |
| `--max-rollouts` | Adaptation rollout attempts | `12` |
| `--width` / `--depth` | Local revision search limits | `2` / `2` |
| `--max-decisions` | Harness decisions per episode | `128` |
| `--max-episode-seconds` | Simulated time per episode | `180` |
| `--rollout-timeout` | Wall-clock timeout per simulator process | `7200` |

Each held-out episode runs separately after harness selection, outside the adaptation rollout budget. Pass `--test-ids ""` to omit held-out testing. Run `bash scripts/run.sh --help` for all options.

### Outputs and resuming

Runs are written outside the source directory, by default to `../AdaHVLA-runs/navila-LH/`:

```text
adaptation.json                   # experiment configuration and adaptation state
selected.json                     # selected candidate and source hash
candidates/C0000/adahvla/          # prototype source snapshot
candidates/C0001/adahvla/          # revised candidate source
checks/                           # source and interface check results
rollouts/R0001/evidence.json       # adaptation rollout evidence
rollouts/R0001/images/             # timestamped, view-labeled observations
rollouts/R0001/output/process.log  # simulator subprocess output
rollouts/T0001/                    # held-out evaluation evidence
```

Rerun with the **same arguments and output directory** to resume. Completed prototype rollouts are reused. Interrupted or failed evaluations can consume an attempt; inspect their evidence before interpreting the results. Use one runner per output directory, and start a new directory when changing episode sets, model configuration, budgets, or fixed evaluation code.

The held-out phase evaluates the selected harness without further code revisions. The online harness still calls the reasoning API while executing held-out tasks. Runtime evidence is generated locally and is not part of the source release.

## Evaluation and validation

The current evaluator reports success when the harness declares completion and the robot's final **3D Euclidean distance** to the target is within the goal radius. It also records termination reasons, actions, trajectory length, and sampled visual evidence. Ground-truth goal coordinates, reference trajectories, and annotated actions are kept out of the online harness input.

This endpoint metric does not verify every intermediate subgoal and is not a geodesic or SPL metric. Sampled images are not a full video. The release does not include the paper's complete experiment configurations or result archive.

Run the offline tests with:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  python -m unittest discover -s tests -v
```

Tests cover harness state transitions, adaptation, candidate loading, runner behavior, action parsing, and CPU controller inference. Controller tests require PyTorch; fake environments and reasoners exercise software contracts. These checks do not establish navigation performance. A fresh simulator installation, the full reasoning API–NaVILA loop, and benchmark-scale adaptation should be validated in the target deployment environment.

## Acknowledgments

This implementation builds on **[NaVILA: Legged Robot Vision-Language-Action Model for Navigation](https://arxiv.org/abs/2412.04453), RSS 2025**. We thank the NaVILA authors for their models, benchmark, environment configuration, and locomotion resources.

| Resource | Upstream project |
| --- | --- |
| VLA model and implementation | [AnjieCheng/NaVILA](https://github.com/AnjieCheng/NaVILA) |
| Navigation benchmark and environment setup | [yang-zj1026/NaVILA-Bench](https://github.com/yang-zj1026/NaVILA-Bench) |
| Simulation scene release | [VLN-CE-Isaac](https://huggingface.co/datasets/Zhaojing/VLN-CE-Isaac), based on [Matterport3D](https://niessner.github.io/Matterport/) |
| Locomotion training | [yang-zj1026/legged-loco](https://github.com/yang-zj1026/legged-loco) |
| Simulator integration | [NaVILA's Isaac Lab fork](https://github.com/yang-zj1026/IsaacLab) |
| Go2 robot assets | NVIDIA Isaac Sim 4.1; sources in [`assets/manifest.json`](assets/manifest.json) |

## License

AdaHVLA's original code is licensed under the [Apache License 2.0](LICENSE).

Third-party code retains its original license and attribution notices, collected in [`THIRD_PARTY_NOTICES.txt`](THIRD_PARTY_NOTICES.txt). Datasets, scene and robot assets, and model weights remain subject to their respective upstream terms; the Apache-2.0 license does not replace those terms.
