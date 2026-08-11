# HomeSafeBench

[![English](https://img.shields.io/badge/Language-English-blue)](README.md)
[![简体中文](https://img.shields.io/badge/语言-简体中文-red)](README-zh.md)
[![Paper](https://img.shields.io/badge/Paper-arXiv-blue)](https://arxiv.org/abs/2509.23690)
[![Data](https://img.shields.io/badge/Data-Hugging%20Face-orange)](https://huggingface.co/datasets/Navinvue/HomeSafeBench)

HomeSafeBench is a benchmark for free-exploration home safety inspection with embodied vision-language models. An agent must actively navigate an interactive 3D home, adjust its viewpoint, and report five categories of hazards from first-person visual observations: fire, electric shock, falling objects, trip hazards, and child safety hazards. Built on VirtualHome, the benchmark contains 1,000 human-validated test tasks.

We also introduce CueBack, a simulator-free offline data construction method that turns the earliest hazard clues along a trajectory into executable visual reasoning supervision.

![HomeSafeBench overview](./assets/main.png)

<p align="center"><em>Figure 1. Overview of HomeSafeBench.</em></p>

![Dataset construction pipeline](./assets/dataset_pipeline.png)

<p align="center"><em>Figure 2. HomeSafeBench dataset construction pipeline.</em></p>

## Usage

### 1. Install VirtualHome

We recommend Linux, Python 3.10, and an isolated Conda environment:

```bash
conda create -n homesafebench python=3.10 -y
conda activate homesafebench
python -m pip install numpy pillow opencv-python requests huggingface_hub
sudo apt-get install -y xvfb
```

Clone VirtualHome and switch to the version used in our experiments:

```bash
mkdir -p third_party
git clone https://github.com/xavierpuigf/virtualhome.git third_party/virtualhome
git -C third_party/virtualhome checkout 58970fd80951c2eaa1af713e0917d1a105353ad8
export VH_ROOT="$PWD/third_party/virtualhome"
```

Download the [VirtualHome Linux v2.3.0 executable](http://virtual-home.org/release/simulator/v2.0/v2.3.0/linux_exec.zip), extract it, and arrange the files as follows:

```text
third_party/virtualhome/unity_vol/linux_exec/
├── linux_exec.v2.3.0.x86_64
├── UnityPlayer.so
└── linux_exec.v2.3.0_Data/
```

Make the executable runnable and start the simulator:

```bash
chmod +x "$VH_ROOT/unity_vol/linux_exec/linux_exec.v2.3.0.x86_64"
xvfb-run --auto-servernum --server-args="-screen 0 640x480x24" \
  "$VH_ROOT/unity_vol/linux_exec/linux_exec.v2.3.0.x86_64" \
  -batchmode -http-port=18188
```

Once the simulator is running, test the connection in another terminal:

```bash
conda activate homesafebench
export VH_ROOT="$PWD/third_party/virtualhome"
python scripts/test_virtualhome.py --port 18188
```

A successful test prints a message similar to the following and saves a 640x480 image to `output/virtualhome_test.png`:

```text
Saved RGB image to output/virtualhome_test.png (640x480)
```

### 2. Prepare the Data

The complete dataset is available at [Navinvue/HomeSafeBench](https://huggingface.co/datasets/Navinvue/HomeSafeBench). Run the following command from the repository root:

```bash
hf download Navinvue/HomeSafeBench \
  data/train/ data/test/ \
  --repo-type dataset \
  --local-dir .
```

After downloading, `data/train/` contains 3,400 training tasks and `data/test/` contains 1,000 test tasks. See [`data/README.md`](data/README.md) for the data format.

### 3. Run Experiments

You need to implement your own model interface. Add your VLM implementation in [`exp/vlm.py`](exp/vlm.py), connect it in `build_vlm` in [`exp/runner.py`](exp/runner.py), and add its registered name to the `--vlm` choices.

With a running VirtualHome instance, you can first execute the default mock run to verify the runner pipeline on one task:

```bash
bash scripts/run_inference.sh
```

The mock output is written separately to `output/mock_results/` and does not represent model performance. To run real inference, specify the registered VLM name:

```bash
VLM_NAME=YOUR_VLM_NAME bash scripts/run_inference.sh
```

Replace `YOUR_VLM_NAME` with the registered name of your implementation. Add `--limit 1` to test a real model on one task. Each task result is written to `output/results/output_<sample>.json`. Run `bash scripts/run_inference.sh --help` for configurable paths, the simulator port, and additional runner arguments.

### 4. Evaluate

The evaluator reads runner outputs and computes precision, recall, and F1. For hazard-level evaluation, connect your LLM in `call_llm_judge` in [`eval/eval_utils.py`](eval/eval_utils.py), then run:

```bash
bash scripts/run_evaluation.sh
```

The script evaluates the first 20 steps with the custom judge and saves results to `output/eval/summary.json`, including the overall metrics and metrics grouped by hazard type. Run `bash scripts/run_evaluation.sh --help` for configurable paths and additional evaluator arguments. To compute category-level metrics without an LLM judge, call `eval/eval.py` directly without `--use_judge`.

## CueBack

The CueBack data release contains 3,158 aligned trajectories in three variants.
They share the same first-person observations, tool calls, and tool feedback,
but differ in the supervision text preceding each action:

- `sft_action.jsonl`: executable actions without reasoning.
- `sft_both.jsonl`: the same actions with rationales generated directly from
  the trajectory prefix, current observation, and action.
- `cueback.jsonl`: the same actions with clue-based reasoning constructed by
  CueBack.

The referenced first-person observations are stored in `images/`. Download all
three variants and their images from the same Hugging Face Dataset:

```bash
hf download Navinvue/HomeSafeBench \
  cueback/sft_action.jsonl cueback/sft_both.jsonl \
  cueback/cueback.jsonl cueback/images/ \
  --repo-type dataset \
  --local-dir .
```

```text
cueback/
├── cueback.jsonl
├── images/
├── sft_action.jsonl
└── sft_both.jsonl
```

## Citation

```bibtex
@misc{yao2026homesafebenchbenchmarkembodiedvisionlanguage,
      title={HomeSafeBench: Benchmarking Embodied Vision-Language Models in Free-Exploration Home Safety Inspection},
      author={Jiashu Yao and Haoyu Wen and Siyuan Gao and Yuhang Guo and Zeming Liu and Heyan Huang},
      year={2026},
      eprint={2509.23690},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2509.23690},
}
```

## License

The code in this repository is released under the [MIT License](LICENSE). The
HomeSafeBench dataset is released under the [CC BY 4.0 License](https://creativecommons.org/licenses/by/4.0/).
