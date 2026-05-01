"""Precompute embeddings of CALVIN instructions.

Encodes the ~17 870 ABC training instructions (or the 5 124 D-only training
instructions, or the 1 087 validation instructions) through a CLIP/BERT
text encoder once, so the data loader doesn't pay the encoder cost during
training. The original implementation ran the encoder one instruction at
a time; this version processes them in batches for a ~B× GPU speedup.
"""
import os
import re
import pickle
from pathlib import Path
from typing import Literal

import tap
import transformers
from tqdm.auto import tqdm
import torch
import numpy as np


TextEncoder = Literal["bert", "clip"]


class Arguments(tap.Tap):
    output: Path
    encoder: TextEncoder = "clip"
    # 16 is the length used by the upstream released artefacts under
    # ``instructions/calvin_task_*_D/`` and by the eval loop's
    # ``text_max_length`` default. The training data loader can fall back
    # to a (B, 53, 512) zero buffer when an embedding is missing, but for
    # parity with the released ckpts use 16 here.
    model_max_length: int = 16
    device: str = "cuda"
    verbose: bool = False
    annotation_path: Path
    batch_size: int = 64


def parse_int(s):
    return int(re.findall(r"\d+", s)[0])


def load_model(encoder: TextEncoder) -> transformers.PreTrainedModel:
    if encoder == "bert":
        model = transformers.BertModel.from_pretrained("bert-base-uncased")
    elif encoder == "clip":
        model = transformers.CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
    else:
        raise ValueError(f"Unexpected encoder {encoder}")
    if not isinstance(model, transformers.PreTrainedModel):
        raise ValueError(f"Unexpected encoder {encoder}")
    return model


def load_tokenizer(encoder: TextEncoder) -> transformers.PreTrainedTokenizer:
    if encoder == "bert":
        tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")
    elif encoder == "clip":
        tokenizer = transformers.CLIPTokenizer.from_pretrained(
            "openai/clip-vit-base-patch32"
        )
    else:
        raise ValueError(f"Unexpected encoder {encoder}")
    if not isinstance(tokenizer, transformers.PreTrainedTokenizer):
        raise ValueError(f"Unexpected encoder {encoder}")
    return tokenizer


if __name__ == "__main__":
    args = Arguments().parse_args()
    print(args)

    annotations = np.load(str(args.annotation_path), allow_pickle=True).item()
    instructions_string = [s + '.' for s in annotations['language']['ann']]

    tokenizer = load_tokenizer(args.encoder)
    tokenizer.model_max_length = args.model_max_length

    model = load_model(args.encoder).to(args.device).eval()

    instructions = {'embeddings': [], 'text': []}

    # Tokenize once with right-padding to the model's max length so each
    # batch is a square (B, L) tensor — saves doing per-instruction padding
    # B times at the encoder boundary.
    bs = max(1, args.batch_size)
    n = len(instructions_string)
    with torch.no_grad():
        for i in tqdm(range(0, n, bs), desc="encoding"):
            chunk = instructions_string[i:i + bs]
            tokens = tokenizer(
                chunk,
                padding="max_length",
                truncation=True,
                max_length=args.model_max_length,
                return_tensors="pt",
            )["input_ids"].to(args.device)
            pred = model(tokens).last_hidden_state  # (B, L, F)
            for j, emb in enumerate(pred.cpu()):
                # Preserve the (1, L, F) layout the data loader expects.
                instructions['embeddings'].append(emb.unsqueeze(0))
                instructions['text'].append(chunk[j])

    os.makedirs(str(args.output.parent), exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(instructions, f)
