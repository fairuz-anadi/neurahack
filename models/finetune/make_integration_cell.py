"""Build ft_integration_cell.py: the one-cell Kaggle handoff for the team notebook (FT family, E_balanced).

    python -m models.finetune.make_integration_cell --synth E:/neurahack/synth_v2/synth_filtered.csv

Embeds (gzip + base64) the synthetic CSV produced by the AutoTrain run and the CC BY 4.0 external snapshot, so the
cell needs no extra Kaggle inputs. Writes the cell to notebooks/ and to ../kaggle_upload/.
"""
from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import io
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = Path(__file__).with_name("integration_cell_template.py")


def pack(raw: bytes) -> tuple[str, str]:
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0, compresslevel=9) as f:
        f.write(raw)
    return base64.b64encode(buf.getvalue()).decode(), hashlib.sha256(raw).hexdigest()[:16]


def build(synth_path: Path, ext_path: Path) -> str:
    syn_b64, syn_sha = pack(synth_path.read_bytes())
    ext_raw = ext_path.read_bytes()
    if ext_path.suffix == ".gz":
        ext_raw = gzip.decompress(ext_raw)
    ext_b64, ext_sha = pack(ext_raw)
    src = TEMPLATE.read_text(encoding="utf-8")
    for k, v in {"__SYNTH_B64__": syn_b64, "__SYNTH_SHA__": syn_sha, "__EXT_B64__": ext_b64, "__EXT_SHA__": ext_sha}.items():
        assert src.count(k) == 1, k
        src = src.replace(k, v)
    compile(src, "ft_integration_cell.py", "exec")
    return src


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth", required=True, help="synth_filtered.csv from the AutoTrain run")
    ap.add_argument("--external", default=str(ROOT / "snapshots" / "external_v2.csv.gz"))
    ap.add_argument("--out", nargs="*", default=[str(ROOT / "notebooks" / "ft_integration_cell.py"),
                                                 str(ROOT.parent / "kaggle_upload" / "ft_integration_cell.py")])
    a = ap.parse_args()
    src = build(Path(a.synth), Path(a.external))
    for o in a.out:
        Path(o).parent.mkdir(parents=True, exist_ok=True)
        Path(o).write_text(src, encoding="utf-8", newline="\n")
        print(f"wrote {o} ({len(src.encode()) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
