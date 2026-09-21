# atml-pa1-beyond-iid

Experimental analysis of distribution shifts in ML, focusing on domain adaptation, domain generalization, and open-set recognition for the ATML PA1 assignment.

## Task 2: PACS unsupervised domain adaptation

Install dependencies with `python -m pip install -r requirements.txt`, then download PACS into the expected layout from the repository root:

```powershell
python -m shared.pacs_protocol --data_root data/PACS
python -m task2.train --config task2/configs/source_only.yaml
python -m task2.train --config task2/configs/dan.yaml
python -m task2.train --config task2/configs/dann.yaml
python -m task2.train --config task2/configs/cdan.yaml
python -m task2.evaluate_final --data_root ./data/PACS
```

Use `--debug` with any training command for one epoch and two optimizer steps. CLI arguments override the YAML configuration. The first run creates the reproducible split file at `splits/pacs_sketch_seed6304.json`.

## Debug mode

Use `--debug` to run each method for one epoch and two optimizer steps:

```powershell
python -m task2.train --config task2/configs/source_only.yaml --debug

python -m task2.train --config task2/configs/dan.yaml --debug

python -m task2.train --config task2/configs/dann.yaml --debug

python -m task2.train --config task2/configs/cdan.yaml --debug
