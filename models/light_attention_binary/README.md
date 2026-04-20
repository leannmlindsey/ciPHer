# light_attention_binary

Light Attention trunk over per-residue embeddings feeding one binary classifier
per serotype (one SimpleMLP + `BCEWithLogitsLoss` per K-type and per O-type).

Adapted from a colleague's PHROG-category classifier (originally trained on 10
PHROG categories); here the same pattern is applied to Klebsiella K/O
serotypes.

## Status
Scaffolding only. Reference source from the original PHROG implementation
will be dropped into `reference/` (kept out of the package import path) so we
can port it into `model.py` / `train.py` / `predict.py` with minimal changes.

## Reference files (to be copied in)
- `reference/modeling_utils.py` — `ConvolutionalAttention` (light attention)
  and `SimpleMLP` (per-class head).
- `reference/training.py` — `TrainModel` Lightning module; see
  `__shared_step` for the BCE-with-logits loss.

Original paths on Delta-AI:
- `/work/hdd/bfzj/hgwak1/models/PPAM-TDM/src/utils/modeling_utils.py`
- `/work/hdd/bfzj/hgwak1/models/PPAM-TDM/src/training.py`
