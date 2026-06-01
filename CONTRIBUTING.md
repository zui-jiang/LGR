# Contributing

Thanks for helping keep this experiment reproducible.

## Development Guidelines

- Keep machine-specific paths, tokens, and cluster endpoints in `.env`.
- Keep experiment scripts executable and runnable from the repository root.
- Document new scripts in `README.md`, including required environment variables.
- Update submodules with normal git submodule workflows instead of copying vendored code into the root repository.

## Before Opening a Pull Request

```bash
git submodule status
bash -n scripts/*.sh
```

If a change touches launch behavior, include the exact command used to validate it and the cluster/runtime assumptions.
