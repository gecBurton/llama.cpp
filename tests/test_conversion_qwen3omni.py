"""Qwen3-Omni mmproj conversion: thinker.visual.* names must map like plain Qwen3-VL ones.

Run from the repository root (deps: requirements/requirements-convert_hf_to_gguf.txt):
    python -m unittest discover -s tests -p 'test_conversion_qwen3omni.py' -v

Regression for the conversion crash on master, where the omni rename produced
'model.visual.blocks.0.attn.proj.bias' at modify_tensors stage:
map_tensor_name raised "Can not map tensor 'model.visual.blocks.0.attn.proj.bias'",
so every Qwen3-Omni mmproj conversion failed. The fix emits the 'visual.' form,
matching the plain Qwen3-VL names the shared pipeline expects.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import torch
from safetensors.torch import save_file

# use the repository's gguf-py like convert_hf_to_gguf.py does: the pip gguf lacks the newest tensor kinds
import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(1, str(ROOT / "gguf-py"))
import gguf
from conversion.qwen3vl import Qwen3OmniMmprojModel


HPARAMS = {
    "model_type": "qwen3_omni_moe",
    "architectures": ["Qwen3OmniMoeForConditionalGeneration"],
    "text_config": {"hidden_size": 2048, "rms_norm_eps": 1e-6},
    "thinker_config": {
        "vision_config": {
            "depth": 1,
            "hidden_size": 8,
            "num_heads": 1,
            "num_position_embeddings": 16,
            "patch_size": 2,
            "spatial_merge_size": 1,
        },
        "audio_config": {
            "d_model": 8,
            "encoder_ffn_dim": 16,
            "encoder_attention_heads": 1,
            "num_mel_bins": 8,
            "layer_norm_eps": 1e-5,
        },
    },
}


class TestQwen3OmniMmprojTensorNames(unittest.TestCase):
    """The checkpoint tensor names come from the public Qwen3-Omni index (curated subset):

    the blkd0 attention output projection ('thinker.visual.blocks.0.attn.proj.bias')
    is the tensor whose mapping raised "Can not map tensor" on plain master.

    The synthetic config carries no merger_list/deepstack tensors, so the test stays
    policy-neutral on the deepstack emission question (see the clef-omni discussion).
    """

    def build(self, path, tensors):
        (path / "config.json").write_text(json.dumps(HPARAMS))
        save_file(tensors, str(path / "model.safetensors"))
        return Qwen3OmniMmprojModel(path, gguf.LlamaFileType.ALL_F32, path / "out.gguf", hparams=HPARAMS)

    def convert(self, tensors):
        with tempfile.TemporaryDirectory() as directory:
            model = self.build(Path(directory), tensors)
            model.prepare_tensors()
            # copy: the tensors are views of the mmapped checkpoint, which must be closed before the directory is removed
            return {name: gguf.LazyNumpyTensor.to_eager(info.tensor).copy() for name, info in model.gguf_writer.tensors[0].items()}

    def test_omni_vision_and_audio_names_map(self):
        tensors = {
            "thinker.visual.patch_embed.proj.weight": torch.ones((4, 1, 2, 2, 2)) / 100,
            "thinker.visual.patch_embed.proj.bias": torch.tensor([0.5]),
            "thinker.visual.pos_embed.weight": torch.ones((8, 8)) / 100,
            "thinker.visual.blocks.0.norm1.weight": torch.ones((8,)),
            "thinker.visual.blocks.0.norm2.weight": torch.ones((8,)),
            "thinker.visual.blocks.0.attn.qkv.weight": torch.ones((24, 8)) / 100,
            "thinker.visual.blocks.0.attn.qkv.bias": torch.tensor([0.5] * 24),
            # the tensor that used to crash the conversion
            "thinker.visual.blocks.0.attn.proj.weight": torch.ones((8, 8)) / 100,
            "thinker.visual.blocks.0.attn.proj.bias": torch.tensor([0.25]),
            "thinker.visual.blocks.0.mlp.linear_fc1.weight": torch.ones((16, 8)) / 100,
            "thinker.visual.blocks.0.mlp.linear_fc2.weight": torch.ones((8, 16)) / 100,
            "thinker.audio_tower.conv2d1.weight": torch.ones((8, 1, 2, 2)) / 100,
            "thinker.audio_tower.layers.0.self_attn.q_proj.weight": torch.ones((8, 8)) / 100,
            "thinker.audio_tower.layers.0.fc1.weight": torch.ones((16, 8)) / 100,
        }

        result = self.convert(tensors)

        for name in result:
            self.assertTrue(name.startswith(("v.", "a.")), f"unexpected mapped name {name}")
        self.assertIn("v.patch_embd.weight", result)
        self.assertIn("v.patch_embd.weight.1", result)
        self.assertIn("v.blk.0.attn_qkv.weight", result)
        self.assertIn("v.blk.0.attn_out.weight", result)
        self.assertIn("v.blk.0.attn_out.bias", result)
        self.assertIn("a.blk.0.attn_q.weight", result)
        # no deepstack tensors: the synthetic config carries no merger_list
        self.assertFalse([n for n in result if n.startswith("v.deepstack.")])


if __name__ == "__main__":
    unittest.main()
