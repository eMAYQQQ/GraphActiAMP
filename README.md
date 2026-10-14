# GraphActiAMP

Run the following commands from the repository root.

## Installation

```bash
python -m pip install -r requirements.txt
```

## Evaluate Stage-2 models

```bash
python stage2/evaluate_stage2_checkpoints.py --device cpu
```

## Evaluate the two-stage cascade

```bash
python stage2/evaluate_two_stage_cascade.py \
  --output-dir outputs/cascade --device -1
```

## Train Stage-1 models

```bash
python stage1/train_stage1.py \
  --variant esm_only --seed 42 --epochs 18 --patience 5 --device cpu \
  --warmup-epochs 2 --output-dir outputs/retrain_stage1_esm

python stage1/train_stage1.py \
  --variant full_residual --seed 42 --epochs 12 --patience 4 --device cpu \
  --warmup-epochs 1 --init-checkpoint outputs/retrain_stage1_esm/best_stage1.pt \
  --freeze-backbone --output-dir outputs/retrain_stage1_final
```

## Train Stage-2 models

```bash
python stage2/train_multimodal_ova.py \
  --split-file stage2/splits/shared_stage2_seed42_split.json \
  --feature-variant esm_only --seed 42 --epochs 100 --batch-size 16 \
  --hidden-dim 384 --dropout 0.35 --lr 0.0002 \
  --sampler weighted --no-asl --minority-boost 3.0 \
  --minority-threshold 0.5 --threshold-grid-start 0.02 \
  --threshold-grid-end 0.85 --threshold-grid-steps 84 \
  --threshold-metric balance --target-sen-spec-gap 0.1 \
  --amp --device 0 --output-dir outputs/retrain_stage2
```
