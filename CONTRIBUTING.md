# Contributing

Please describe the problem, the affected experiment setting, and the validation performed.

- Keep paths, service endpoints, and credentials in a local `.env`; never commit secrets, weights, datasets, or generated outputs.
- `slime_tree/` and `sg_tree/` are vendored source trees. Edit them directly and keep upstream licenses and notices intact.
- Keep changes to training objectives separate from changes to launch infrastructure. Document new experimental settings and their usage.
- Avoid broad formatting or dependency upgrades across the vendored code.

Before proposing a change:

```bash
bash -n scripts/train_lgr.sh scripts/exp258-distill-1.5B-v3.sh scripts/models/*.sh
bash scripts/train_lgr.sh --dry-run
git diff --check
```

For serving changes, run `sg_tree/test_tree_attention.py` against a teacher using the local SGLang fork. For training changes, report the hardware, dependency versions, launch configuration, and a short training validation. A dry run checks command construction; it does not validate GPU execution or reproduce paper results.
