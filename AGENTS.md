# AGENTS.md — rules for working in this repository

These rules are non-negotiable for humans and agents alike.

1. Versions come from `runtime.lock.json`. Do not build or pin anything else without updating the lock in the same change. Current pins: vLLM v0.30.0rc1 (a00a3544b93e), exllamav3 v1.5.0 (0740edc2), torch 2.13.0+cu130, CUDA 13.0, TORCH_CUDA_ARCH_LIST="12.0;12.1", flashinfer 0.6.18.post1, transformers 5.17.x.

2. The target platform is GB10 / DGX Spark: aarch64, SM121, CUDA 13, ATS unified memory (~121.7 GiB usable). Anything that assumes x86, discrete VRAM, or sub-128 thread-block semantics must be flagged.

3. License firewall. Publishable code in this repository descends only from: vLLM (Apache-2.0), turboderp-org/exllamav3 (MIT), and any file recorded in `docs/provenance.md`. AGPL-licensed third-party engines and community recipe repositories may be read and run privately for comparison, but must never be copied, vendored, or adapted into this repository. New files that adapt third-party MIT code must be added to `docs/provenance.md` in the same commit.

4. Patches under `patches/vllm/` must be anchored (exact context lines) and fail closed if an anchor does not match. Never write a patch that silently skips.

5. Every claim of "works" needs a receipt: the exact command, the runtime identity (vLLM commit, plugin commit, kernel arch, model revision), and the observed output. Evidence directories use sanitized hostnames.

6. Do not run GB10 work on developer workstations. Anything requiring the GPU, aarch64, or >32 GiB must be gated behind `scripts/bootstrap_cluster.sh` and executed on the cluster fleet.

7. ATS/zero-copy placement decisions must always leave a memory receipt (weights resident, tables non-resident, reserve discipline) — see `placement.py` receipts.

8. When in doubt about tier-1 facts (vLLM build flags, exllamav3 kernel behavior), check the pinned sources in `vendored/` and the upstream docs before inventing anything.

9. Keep the fork delta small. Prefer out-of-tree plugin code, then anchored patches, then (only if unavoidable) hard forks of vLLM files — and record the reason in `patches/vllm/README.md`.

10. Zero-dependency Python: scripts in `scripts/` use only the Python standard library unless the target venv already has the dependency.
