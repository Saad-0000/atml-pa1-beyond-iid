# atml-pa1-beyond-iid

Experimental analysis of distribution shifts in ML, focusing on domain adaptation, domain generalization, and open-set recognition for the ATML PA1 assignment.

## Task 2: PACS Unsupervised Domain Adaptation

Task 2 evaluates domain adaptation on PACS using **Photo, Art Painting, and Cartoon** as source domains and **Sketch** as the target domain.

### Setup

From the repository root:

```powershell
python -m pip install -r requirements.txt
python -m shared.pacs_protocol --data_root data/PACS
```

The first command creates the reproducible split:

```text
splits/pacs_sketch_seed6304.json
```

### Debug Run

Use `--debug` for a quick pipeline test. It runs training for **1 epoch and 2 optimizer steps** and uses a small evaluation subset.

```powershell
python -m task2.train --config task2/configs/source_only.yaml --debug
python -m task2.train --config task2/configs/dan.yaml --debug
python -m task2.train --config task2/configs/dann.yaml --debug
python -m task2.train --config task2/configs/cdan.yaml --debug

python -m task2.evaluation.evaluate_final --data_root ./data/PACS --debug
```

### Full Run

```powershell
python -m task2.train --config task2/configs/source_only.yaml
python -m task2.train --config task2/configs/dan.yaml
python -m task2.train --config task2/configs/dann.yaml
python -m task2.train --config task2/configs/cdan.yaml

python -m task2.evaluation.evaluate_final --data_root ./data/PACS
```

Checkpoints are saved in:

```text
task2/results/checkpoints/
```

Final evaluation results are saved in:

```text
task2/results/
```

CLI arguments override values specified in the YAML configuration files.

### Controlled Studies

DAN evaluates:

```text
λ = 0.1, 1.0, 10.0
```

DANN evaluates:

```text
α = 0.25, 0.5, 1.0
```

Run the controlled studies after training the corresponding checkpoints:

```powershell
# DAN
python -m task2.train --config task2/configs/dan.yaml --param_val 0.1
python -m task2.train --config task2/configs/dan.yaml --param_val 1.0
python -m task2.train --config task2/configs/dan.yaml --param_val 10.0
python -m task2.evaluation.plot_controlled_study --study_type dan --data_root ./data/PACS

# DANN
python -m task2.train --config task2/configs/dann.yaml --param_val 0.25
python -m task2.train --config task2/configs/dann.yaml --param_val 0.5
python -m task2.train --config task2/configs/dann.yaml --param_val 1.0
python -m task2.evaluation.plot_controlled_study --study_type dann --data_root ./data/PACS
```

Use `--debug` with the controlled-study evaluation commands for a quick pipeline test.
