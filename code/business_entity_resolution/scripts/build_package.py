"""Assemble <team>_submission.zip exactly as the organisers specify:

<team>_submission.zip
├── output/{matching_results.tsv, candidate_pairs.tsv}
├── code/business_entity_resolution/{src/, scripts/, kaggle/, tests/, README.md, requirements.txt}
└── Documentation_template.md

    python scripts/build_package.py --team MyTeam --version v6
The chosen version's TSVs are taken from artifacts/submissions/<version>/ and validated
(every test S1 once, no duplicates, S2/S3 ids only, matches ⊆ candidates) before zipping."""
import argparse
import shutil
import zipfile
from pathlib import Path

from ber import config

PKG = config.PKG_ROOT
ROOT = config.REPO_ROOT
CODE_PARTS = ["src", "scripts", "kaggle", "tests", "README.md", "requirements.txt", "pytest.ini"]


def check_outputs(match_p: Path, cand_p: Path) -> None:
    s1 = [l.split("\t", 1)[0] for l in open(config.DATA_DIR / "test" / "test_source1.tsv", encoding="utf-8")][1:]
    need = set(x for x in s1 if x)

    def read(p):
        rows = {}
        with open(p, encoding="utf-8") as f:
            header = f.readline().rstrip("\n").split("\t")
            assert header[0] == "source1_entity_id", f"bad header in {p.name}"
            for line in f:
                a, _, b = line.rstrip("\n").partition("\t")
                assert a not in rows, f"duplicate S1 row {a} in {p.name}"
                ids = [x for x in b.split(",") if x]
                assert len(ids) == len(set(ids)), f"duplicate ids in a list ({a}) in {p.name}"
                assert all(x.startswith(("S2-", "S3-")) for x in ids), f"non S2/S3 id in {p.name}"
                rows[a] = set(ids)
        assert set(rows) == need, f"{p.name}: S1 rows do not match the test set"
        return rows
    m, c = read(match_p), read(cand_p)
    bad = sum(1 for s, ids in m.items() if not ids <= c[s])
    assert bad == 0, f"{bad} S1 rows have matches outside candidate_pairs.tsv"
    print(f"outputs OK: {len(m):,} S1 rows; {sum(map(len, m.values())):,} matches; "
          f"{sum(map(len, c.values())) / len(c):.2f} candidates/S1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", required=True)
    ap.add_argument("--version", required=True, help="folder under artifacts/submissions/")
    a = ap.parse_args()
    src = config.ARTIFACT_DIR / "submissions" / a.version
    check_outputs(src / "matching_results.tsv", src / "candidate_pairs.tsv")
    stage = ROOT / "deliverables" / f"{a.team}_submission"
    if stage.exists():
        shutil.rmtree(stage)
    (stage / "output").mkdir(parents=True)
    for f in ("matching_results.tsv", "candidate_pairs.tsv"):
        shutil.copy(src / f, stage / "output" / f)
    dst = stage / "code" / "business_entity_resolution"
    dst.mkdir(parents=True)
    ignore = shutil.ignore_patterns("__pycache__", ".pytest_cache", "*.pyc")
    for part in CODE_PARTS:
        p = PKG / part
        if p.is_dir():
            shutil.copytree(p, dst / part, ignore=ignore)
        elif p.exists():
            shutil.copy(p, dst / part)
    shutil.copy(ROOT / "deliverables" / "Documentation_template.md", stage / "Documentation_template.md")
    zpath = ROOT / "deliverables" / f"{a.team}_submission.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for f in sorted(stage.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(stage))  # contents at zip root, as specified
    print(f"wrote {zpath} ({zpath.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
