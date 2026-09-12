# CLI examples

Run after installation into an activated environment:

```bash
dehcode doctor --json
dehcode models --json
dehcode recommend --json
dehcode install wan-1.3b --dependencies
dehcode run wan-1.3b --prompt "A cat walking on grass" --seed 42 --output outputs/cat.mp4
```

Reduced GPU smoke test (not a quality demonstration):

```bash
dehcode run wan-1.3b --prompt "A cat walking on grass" --width 256 --height 256 --frames 17 --steps 2 --output outputs/smoke.mp4
```

The final two commands require downloading actual weights and compatible NVIDIA hardware. They have not been GPU-validated in this project yet.
