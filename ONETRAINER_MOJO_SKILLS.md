# OneTrainer -> Mojo Agent Skill Pack

These repo-local skills are the standard workflow for Codex and Claude Code
when bringing a OneTrainer model into mojodiffusion.

Canonical skill files are mirrored under both `.claude/skills/` and
`.codex/skills/`.

Use in this order:

1. `.claude/skills/ot-mojo-model-intake/SKILL.md`
2. `.claude/skills/mojo-model-runtime-port/SKILL.md`
3. `.claude/skills/ot-mojo-training-port/SKILL.md`
4. `.claude/skills/ot-mojo-sampler-port/SKILL.md`
5. `.claude/skills/ot-mojo-parity-gates/SKILL.md`
6. `.claude/skills/ot-mojo-production-use/SKILL.md`
7. `.claude/skills/ot-mojo-agent-handoff/SKILL.md`

Codex mirror:

1. `.codex/skills/ot-mojo-model-intake/SKILL.md`
2. `.codex/skills/mojo-model-runtime-port/SKILL.md`
3. `.codex/skills/ot-mojo-training-port/SKILL.md`
4. `.codex/skills/ot-mojo-sampler-port/SKILL.md`
5. `.codex/skills/ot-mojo-parity-gates/SKILL.md`
6. `.codex/skills/ot-mojo-production-use/SKILL.md`
7. `.codex/skills/ot-mojo-agent-handoff/SKILL.md`

Core rule: OneTrainer is the reference. Mojo product code must stay
Mojo-native. Python is allowed for oracle dumps, static guards, measurement
wrappers, and tooling only.

For any new model that needs inference, sampling, video generation, or a
Mojo-only inference UI, build the shared Mojo runtime first. The trainer should
reuse the same loader, block math, conditioning, scheduler, VAE/decode, dtype,
offload, LoRA target, and artifact contracts instead of carrying a second stack.

Before claiming a model is ready, run the parity/readiness gates and report the
actual evidence level: compile, smoke, artifact consumer, loss bridge,
state-init, update-bearing, Mojo replay, or production parity.
