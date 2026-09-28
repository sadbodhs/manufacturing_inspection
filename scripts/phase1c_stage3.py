#!/usr/bin/env python3
"""Phase 1c: stage 3 on TensorRT - what an open-vocabulary detector costs.

Stage 3 names what stage 2 flagged ("scratch", "dent", "missing screw") from a
text prompt, so a new defect type is a new phrase, not a new model. The
candidate is Grounding DINO (tiny: Swin-T image backbone + BERT-base text
encoder, 900 queries), built from its default config with random weights.

Three questions:

  1. Does it convert to TensorRT at all? It has multi-scale deformable
     attention (grid_sample), a two-stage top-k query selection and
     text-conditioned masks. If it does not, that failure is the result, and
     the fallbacks are YOLO-World-L (known to convert) and timing it outside
     TensorRT.
  2. What does a phrase cost? 1, 10 and 80 phrases of three tokens each become
     token sequences of 5, 32 and 242 (the limit is 256).
  3. Should the text encoding be cached? On a line the prompt set is fixed, so
     BERT's output can be computed once. `cached` replaces the text encoder with
     its precomputed output (a graph constant); `live` runs BERT per request.

Input: a flagged crop at 384, or the full frame at 800. Batch 1: stage 3 is
rare and latency-bound (Phase 0 asks whether batching would help it).

Token ids are fixed and realistic in structure: [CLS], then per phrase two word
ids and '.', then [SEP]. The masks Grounding DINO derives from '.' positions are
therefore traced as constants, which matches a fixed prompt set.

PREDICTIONS (written 2026-09-27, before any export was attempted)

  P1  The export and build succeed, but only after work: at least one op needs a
      workaround (most likely deformable attention's grid_sample path or the
      top-k query selection). The conversion fits in the 1-hour cap.
  P2  At 800 with 10 phrases, live text: ~15-25 ms. At a 384 crop: ~5-9 ms,
      not 4.3x less, because the 900-query decoder does not shrink with the image.
  P3  Phrase count is cheap until it is large: 1 -> 10 phrases adds < 10%;
      80 phrases (242 tokens) adds 20-40%, through BERT and the text
      cross-attention in every encoder and decoder layer.
  P4  Caching the text saves BERT's share: ~1 ms at 10 phrases, ~2-3 ms at 80,
      more as a fraction at 384 than at 800.

Usage (inside the container): python3 phase1c_stage3.py <work> <raw.tsv> <summary.tsv>
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import trt_bench as tb  # noqa: E402

CLS, SEP, DOT = 101, 102, 1012


def token_ids(phrases):
    g = torch.Generator().manual_seed(phrases)
    ids = [CLS]
    for _ in range(phrases):
        ids += torch.randint(2000, 20000, (2,), generator=g).tolist() + [DOT]
    ids.append(SEP)
    return torch.tensor([ids])


class ConstText(torch.nn.Module):
    """Stands in for BERT once its output has been computed for a fixed prompt."""
    def __init__(self, hidden):
        super().__init__()
        self.register_buffer("hidden", hidden)

    def forward(self, *args, **kwargs):
        from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
        return BaseModelOutputWithPoolingAndCrossAttentions(last_hidden_state=self.hidden)


class GDino(torch.nn.Module):
    """Conversion workarounds, each logged in the results page:

    1. The text masks come from generate_masks_with_special_tokens_and_transfer_map,
       which uses torch.isin: no ONNX export at opset 17. It depends only on the
       token ids, which are fixed for a prompt set, so it is computed once here and
       the traced graph receives the masks as constants.
    """
    def __init__(self, phrases, cached):
        super().__init__()
        from transformers import GroundingDinoConfig, GroundingDinoForObjectDetection
        from transformers.models.grounding_dino import modeling_grounding_dino as gd
        self.m = GroundingDinoForObjectDetection(GroundingDinoConfig()).eval()
        self.register_buffer("ids", token_ids(phrases))
        self.register_buffer("mask", torch.ones_like(self.ids))
        self.register_buffer("types", torch.zeros_like(self.ids))
        tsam, pos = gd.generate_masks_with_special_tokens_and_transfer_map(self.ids)   # workaround 1
        self.register_buffer("tsam", tsam)
        self.register_buffer("pos", pos)
        gd.generate_masks_with_special_tokens_and_transfer_map = lambda _ids: (self.tsam, self.pos)
        if cached:
            with torch.no_grad():
                out = self.m.model.text_backbone(input_ids=self.ids, attention_mask=self.mask,
                                                 token_type_ids=self.types)
            self.m.model.text_backbone = ConstText(out.last_hidden_state)

    def forward(self, x):
        o = self.m(pixel_values=x, input_ids=self.ids, attention_mask=self.mask,
                   token_type_ids=self.types)
        return o.logits, o.pred_boxes


def exporter(phrases, cached, size):
    def export(onnx_path):
        m = GDino(phrases, cached).eval()
        with torch.no_grad():
            torch.onnx.export(m, (torch.rand(1, 3, size, size),), onnx_path,
                              input_names=["images"], output_names=["logits", "boxes"],
                              opset_version=17, dynamo=False)
        return sum(p.numel() for p in m.parameters())
    return export


def engines():
    out = []
    for size in (384, 800):
        for phrases in (1, 10, 80):
            for cached in (False, True):
                name = "gdino_t_p%d_%s" % (phrases, "cached" if cached else "live")
                out.append(tb.Engine(name, size, 1, exporter(phrases, cached, size)))
    return out


def main():
    work, raw_tsv, sum_tsv = sys.argv[1:4]
    torch.manual_seed(0)
    only = sys.argv[4:]  # optional: engine tags, for the conversion attempt on one engine
    es = [e for e in engines() if not only or e.tag in only]
    tb.run(es, work, raw_tsv, sum_tsv)


if __name__ == "__main__":
    main()
