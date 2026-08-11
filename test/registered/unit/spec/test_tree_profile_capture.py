import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from sglang.srt.environ import envs
from sglang.srt.speculative import tree_profile_capture
from sglang.srt.speculative.tree_profile_capture import TreeProfileCapture
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestTreeProfileCapture(CustomTestCase):
    @staticmethod
    def _inputs(batch_size=4, num_draft_tokens=6, vocab_size=16):
        probs = torch.full(
            (batch_size, num_draft_tokens, vocab_size),
            1.0 / vocab_size,
            dtype=torch.float32,
        )
        topology = torch.arange(num_draft_tokens).expand(batch_size, -1)
        verify_input = SimpleNamespace(
            retrieve_index=topology,
            retrieve_next_token=topology,
            retrieve_next_sibling=topology,
            tree_topk=2,
            max_tree_depth=4,
        )
        sampling_info = SimpleNamespace(
            top_ks=torch.full((batch_size, 1), vocab_size, dtype=torch.int64),
            top_ps=torch.full((batch_size, 1), 0.5, dtype=torch.float32),
            temperatures=torch.ones((batch_size, 1), dtype=torch.float32),
        )
        return probs, topology, verify_input, sampling_info

    def test_bounded_capture_and_metadata(self):
        with tempfile.TemporaryDirectory() as output_dir:
            capture = TreeProfileCapture(output_dir, max_rows=24)
            probs, candidates, verify_input, sampling_info = self._inputs()
            coins = torch.zeros((4, 6), dtype=torch.float32)
            final_coins = torch.zeros(4, dtype=torch.float32)
            accept_token_num = torch.tensor([1, 2, 3, 4], dtype=torch.int32)

            for _ in range(3):
                pending = capture.begin(
                    softmax_probs=probs,
                    candidates=candidates,
                    verify_input=verify_input,
                    sampling_info=sampling_info,
                )
                capture.set_top_p_input(pending, probs)
                capture.finish(
                    pending,
                    renormalized_probs=probs,
                    coins=coins,
                    final_coins=final_coins,
                    accept_token_num=accept_token_num,
                )

            paths = sorted(Path(output_dir).glob("*.pt"))
            self.assertEqual(len(paths), 2)
            records = [torch.load(path, weights_only=True) for path in paths]
            self.assertEqual(sum(x["metadata"]["captured_rows"] for x in records), 24)
            self.assertTrue(
                all(x["metadata"]["nucleus_size"] == [9] * 12 for x in records)
            )
            self.assertTrue(
                all(not any(x["metadata"]["prefix_overflow"]) for x in records)
            )

            manifest = [
                json.loads(line)
                for line in (Path(output_dir) / "manifest.jsonl")
                .read_text()
                .splitlines()
            ]
            self.assertEqual([row["call_index"] for row in manifest], [0, 1])
            self.assertEqual([row["batch_bucket"] for row in manifest], [4, 4])

    def test_arm_file_delays_capture_initialization(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "output"
            arm_file = Path(temp_dir) / "capture.arm"
            tree_profile_capture._CAPTURE = None
            try:
                with (
                    envs.SGLANG_DEBUG_SPEC_PROBS_CAPTURE_DIR.override(str(output_dir)),
                    envs.SGLANG_DEBUG_SPEC_PROBS_CAPTURE_ARM_FILE.override(
                        str(arm_file)
                    ),
                ):
                    self.assertIsNone(tree_profile_capture.get_tree_profile_capture())
                    arm_file.touch()
                    self.assertIsNotNone(
                        tree_profile_capture.get_tree_profile_capture()
                    )
            finally:
                tree_profile_capture._CAPTURE = None


if __name__ == "__main__":
    unittest.main()
