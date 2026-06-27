from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from evaluation.plot_xp_csv_heatmap import format_col_label, format_row_label, generate_heatmap_from_csv, iter_csvs


def load_bundle_module():
    script = Path(__file__).resolve().parents[1] / "scripts/build_xp_heatmap_bundle.py"
    spec = importlib.util.spec_from_file_location("build_xp_heatmap_bundle", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_matrix_csv(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "---,br_for_ippo_mlp_0 (0),br_for_human_proxy (0)",
                "ippo_mlp (0),1.00,0.40",
                "human_proxy,0.55,1.00",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


class XPHeatmapBundleTests(unittest.TestCase):
    def test_formats_paper_agent_labels(self):
        self.assertEqual(format_row_label("ippo_mlp (0)"), "IPPO (0)")
        self.assertEqual(format_row_label("human_proxy"), "Human")
        self.assertEqual(format_col_label("br_for_comedi_1_0 (0)"), "BR CoMeDi-br (1, 0)")
        self.assertEqual(format_col_label("br_for_human_proxy (0)"), "BR Human (0)")

    def test_generate_heatmap_from_csv_writes_pdf(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            csv_path = tmp_path / "toy__returned_episode_returns_mean_brcolmax_normalized=True.csv"
            out_path = tmp_path / "toy.pdf"
            write_matrix_csv(csv_path)

            generated = generate_heatmap_from_csv(csv_path, title="Toy", out_path=out_path, vmin=0.0, vmax=1.0)

            self.assertEqual(generated, out_path)
            self.assertTrue(out_path.exists())
            self.assertTrue(out_path.read_bytes().startswith(b"%PDF"))

    def test_iter_csvs_skips_tidy_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            matrix_csv = tmp_path / "toy__returned_episode_returns_mean_brcolmax_normalized=True.csv"
            tidy_csv = tmp_path / "toy__returned_episode_returns_mean_brcolmax_normalized=True_tidy.csv"
            write_matrix_csv(matrix_csv)
            tidy_csv.write_text("row_agent,col_agent,mean\n", encoding="utf-8")

            self.assertEqual(list(iter_csvs(tmp_path)), [matrix_csv])

    def test_build_bundle_validates_and_zips_matrix_csvs(self):
        module = load_bundle_module()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            csv_dir = tmp_path / "csvs"
            out_dir = tmp_path / "bundle"
            csv_dir.mkdir()
            matrix_csv = csv_dir / "toy__returned_episode_returns_mean_brcolmax_normalized=True.csv"
            write_matrix_csv(matrix_csv)
            (csv_dir / "toy__returned_episode_returns_mean_reconstructed_raw.csv").write_text(
                matrix_csv.read_text(encoding="utf-8"),
                encoding="utf-8",
            )

            zip_path = module.build_bundle(
                csv_dir,
                out_dir,
                expected_pdf_count=1,
                max_tolerance=1e-6,
                make_zip=True,
            )

            self.assertEqual(zip_path, out_dir.with_suffix(".zip"))
            self.assertTrue((out_dir / "pdfs/toy__xp_return_brcolmax_heatmap.pdf").exists())
            manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertAlmostEqual(manifest["outputs"]["toy"]["max_after_normalization"], 1.0)
            with zipfile.ZipFile(zip_path) as archive:
                names = archive.namelist()
            self.assertIn("bundle/pdfs/toy__xp_return_brcolmax_heatmap.pdf", names)
            self.assertFalse(any(".DS_Store" in name for name in names))


if __name__ == "__main__":
    unittest.main()
