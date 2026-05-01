# Lie Diffuser Actor

Official open-source release for *The Lie We Tell: Correcting the Euclidean Fallacy in Vision Language Action Policies via Score Matching on Tangent Space* (ICML 2026).

**🌐 Project page: [tars3017.github.io/lie-diffuser-actor](https://tars3017.github.io/lie-diffuser-actor/)**

## Quickstart

Two benchmarks, each in its own subdirectory with a self-contained README:

- [`calvin/README.md`](calvin/README.md) — CALVIN ABC→D / ABCD→D long-horizon eval (paper Table 1).
- [`openvla_oft/README.md`](openvla_oft/README.md) — LIBERO-10 (LIBERO-Long) rebuttal experiments on top of OpenVLA-OFT.

## References

This codebase builds directly on three upstream projects; please cite them alongside our paper if you use the corresponding subsystems.

- **[`nickgkan/3d_diffuser_actor`](https://github.com/nickgkan/3d_diffuser_actor)** — the diffusion-policy backbone the CALVIN sibling extends. The `Encoder`, `DiffusionHead`, and the cross-attention-into-context architecture in `calvin/lda/encoder/` and `calvin/lda/model/diffuser_actor.py` originate here, with the Lie-tangent prediction head and ablation flags layered on top.
- **[`pithreeone/liepose_pytorch`](https://github.com/pithreeone/liepose_pytorch)** — the score-matching-on-Lie-groups machinery (`NormalSE3`, `NormalSO3_Flat`, `PowerNoiseSchedule`, `lie_metrics`, `ops`) under `calvin/lda/diffusion/lie/` is adapted from this reference. Our additions are the integration with the diffusion-policy backbone and the Euclidean-vs-Lie ablation switch.
- **[`moojink/openvla-oft`](https://github.com/moojink/openvla-oft)** — the parallel-decoding + continuous-action-head fine-tuning recipe vendored under `openvla_oft/{experiments,prismatic,vla-scripts}/`. Our additions are the SE(3) score-matching action head and the YAML-driven wrapper (`openvla_oft/lda_oft/`) that toggles between baseline / Euclidean SM / Lie SM variants.

## Citation

```bibtex
@inproceedings{lda2026,
  title={The Lie We Tell: Correcting the Euclidean Fallacy in Vision Language Action Policies via Score Matching on Tangent Space},
  author={Anonymous Authors},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2026}
}
```

## License

MIT. See the [LICENSE](LICENSE) file.
