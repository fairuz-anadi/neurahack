import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))


def test_build_notebook_never_writes_finetune_notebook(tmp_path, monkeypatch):
    import build_notebook as bn

    monkeypatch.setattr(bn, "NB_DIR", tmp_path)
    sentinel = tmp_path / "04_finetune.ipynb"
    sentinel.write_text("SAMPRITY", encoding="utf-8")
    bn.main()
    assert sentinel.read_text(encoding="utf-8") == "SAMPRITY"
    assert (tmp_path / "05_final_pipeline.ipynb").exists()
    assert (tmp_path / "03_embeddings.ipynb").exists()
