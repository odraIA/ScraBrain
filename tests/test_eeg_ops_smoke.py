from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class EEGOpsSmokeTests(unittest.TestCase):
    def test_eeg_dataset_loading_synthetic_openneuro(self) -> None:
        from scripts import smoke_eeg_word_datasets

        if smoke_eeg_word_datasets.MISSING_DEPENDENCY:
            self.skipTest(f"missing optional dependency: {smoke_eeg_word_datasets.MISSING_DEPENDENCY}")

        with tempfile.TemporaryDirectory() as tmp:
            result = smoke_eeg_word_datasets.smoke_openneuro(Path(tmp))
        self.assertEqual(result["ds004408"]["segments"], 1)
        self.assertIn("listening", result["ds007808"]["tasks"])

    def test_bids_eeg_channel_scanner_uses_channels_tsv(self) -> None:
        try:
            from brainstorm.data.eeg_word_aligned_dataset import scan_bids_eeg_channel_counts
        except ModuleNotFoundError as exc:
            self.skipTest(f"missing optional dependency: {exc.name}")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "sub-01" / "eeg" / "sub-01_task-delong_run-01_eeg.vhdr"
            raw.parent.mkdir(parents=True)
            raw.touch()
            raw.with_name("sub-01_task-delong_run-01_channels.tsv").write_text(
                "\n".join(
                    [
                        "name\ttype",
                        "Fp1\tEEG",
                        "Fp2\tEEG",
                        "VEOG\tEOG",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            counts = scan_bids_eeg_channel_counts(root, tasks=["delong"])

        self.assertEqual(len(counts), 1)
        self.assertEqual(counts[0].n_channels, 2)
        self.assertEqual(counts[0].method, "channels.tsv")

    def test_openneuro_textgrid_preflight_reports_unmaterialized_sidecars(self) -> None:
        try:
            from brainstorm.data.eeg_word_aligned_dataset import OpenNeuroEEGWordAlignedDataset
        except ModuleNotFoundError as exc:
            self.skipTest(f"missing optional dependency: {exc.name}")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "sub-001" / "eeg" / "sub-001_task-listening_run-01_eeg.vhdr"
            raw.parent.mkdir(parents=True)
            raw.touch()
            stimuli = root / "stimuli"
            stimuli.mkdir()
            (stimuli / "audio01.TextGrid").symlink_to(root / ".git" / "annex" / "missing-textgrid")

            with self.assertRaisesRegex(FileNotFoundError, "No materialized TextGrid sidecars"):
                OpenNeuroEEGWordAlignedDataset(
                    data_root=str(root),
                    dataset_name="openneuro_ds004408",
                    task_mode="listening",
                    tasks=["listening"],
                )

    def test_hydra_sweep_config_loading(self) -> None:
        from scripts.make_eeg_sweep_plan import build_plan, load_config

        cfg = load_config(REPO_ROOT / "configs" / "eeg_sweep.yaml")
        self.assertIn("base_configs", cfg)
        plan = build_plan(REPO_ROOT / "configs" / "eeg_sweep.yaml", task_mode_filter="reading", tokenizer_filter="biocodec")
        self.assertGreaterEqual(len(plan), 1)
        self.assertEqual(plan[0]["task_mode"], "reading")

    def test_tokenizer_factory_metadata(self) -> None:
        try:
            import torch  # noqa: F401
        except ModuleNotFoundError as exc:
            self.skipTest(f"missing optional dependency: {exc.name}")

        from brainstorm.neuro_tokenizers.factory import load_neuro_tokenizer

        tiny_cfg = REPO_ROOT / "brainstorm" / "neuro_tokenizers" / "tiny" / "model_cfg.json"
        tiny_ckpt = REPO_ROOT / "brainstorm" / "neuro_tokenizers" / "tiny" / "BrainOmni.pt"
        if not tiny_cfg.exists() or not tiny_ckpt.exists():
            self.skipTest("BrainOmni tiny metadata files are not available")
        tokenizer = load_neuro_tokenizer("brainomni_tiny")
        self.assertGreater(tokenizer.downsample_ratio, 0)
        self.assertGreater(tokenizer.n_q, 0)
        self.assertGreater(tokenizer.vocab_size, 0)

    def test_monitor_status_empty_workspace(self) -> None:
        import monitor_server

        old_base = monitor_server.BASE_DIR
        old_logs = monitor_server.LOGS_DIR
        old_results = monitor_server.RESULTS_DIR
        old_ckpt = monitor_server.CKPT_DIR
        old_promotions = monitor_server.PROMOTIONS_DIR
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            for name in ("logs", "results", "checkpoints", "promotions"):
                (base / name).mkdir()
            monitor_server.BASE_DIR = base
            monitor_server.LOGS_DIR = base / "logs"
            monitor_server.RESULTS_DIR = base / "results"
            monitor_server.CKPT_DIR = base / "checkpoints"
            monitor_server.PROMOTIONS_DIR = base / "promotions"
            status = monitor_server.get_sweep_status()
            chained = monitor_server.get_chained_status()
        monitor_server.BASE_DIR = old_base
        monitor_server.LOGS_DIR = old_logs
        monitor_server.RESULTS_DIR = old_results
        monitor_server.CKPT_DIR = old_ckpt
        monitor_server.PROMOTIONS_DIR = old_promotions

        self.assertIn("experiments", status)
        self.assertIn("current_stage", chained)

    def test_chained_sweep_dry_run(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "run_eeg_chained_sweep.py"),
                "--dry-run",
                "--limit",
                "1",
            ],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("Stage 1", completed.stdout)


if __name__ == "__main__":
    unittest.main()
