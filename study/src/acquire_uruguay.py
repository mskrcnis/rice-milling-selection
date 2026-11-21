"""Acquire the original INIA files through public Dryad or its exact Zenodo mirror."""

from pathlib import Path
import argparse, datetime, hashlib, json, urllib.request
from concurrent.futures import ThreadPoolExecutor

B = Path(__file__).resolve().parents[1]
R = B / "raw/uruguay"
A = B / "audit"
FILES = {
    "README.md": (
        "2465930",
        "da3cee8cee3121dc1f90c8b86604a070eda7bc4e2215b6106329235e0488ea52",
    ),
    "Phenotypes.txt": (
        "2465928",
        "8e587890c3d5d479f4f43b7202eb365c4b866f4fe6f15c5d709ff2bc3eb33d0b",
    ),
    "Trials.txt": (
        "2465926",
        "2a75b0b5488fd243093bb4f6178e31142c0f42e8e739f58130782529f7dc103a",
    ),
    "Lines.txt": (
        "2465924",
        "644417788d35a4e236e084ecc24a230c9ed13c5896befdd052409470e6fed4f6",
    ),
}
META = {
    "dryad_version.json": "https://datadryad.org/api/v2/versions/248051",
    "dryad_files.json": "https://datadryad.org/api/v2/versions/248051/files",
    "zenodo_record.json": "https://zenodo.org/api/records/8222856",
}


def get(url):
    req = urllib.request.Request(
        url, headers={"User-Agent": "Rice-milling-reproducibility/1.0"}
    )
    with urllib.request.urlopen(req, timeout=45) as f:
        return f.read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args()
    R.mkdir(parents=True, exist_ok=True)
    old = A / "uruguay_source_manifest.json"
    previous = (
        {r.get("name"): r for r in json.loads(old.read_text())} if old.exists() else {}
    )

    def acquire(item):
        name, (fid, expected) = item
        p = R / name
        public = f"https://datadryad.org/downloads/file_stream/{fid}"
        mirror = f"https://zenodo.org/api/records/8222856/files/{name}/content"
        row = {
            "name": name,
            "path": str(p.relative_to(B)),
            "dataset_doi": "10.5061/dryad.x69p8czn8",
            "dryad_version_id": 248051,
            "public_url": public,
            "mirror_url": mirror,
            "url": mirror,
            "license": "CC0-1.0",
            "redistribution": "CC0 permits reuse; README requests optional author contact, no contact sent",
            "expected_sha256": expected,
        }
        if p.exists():
            sha = hashlib.sha256(p.read_bytes()).hexdigest()
            if sha != expected:
                raise ValueError("Cached source hash mismatch: " + name)
            prior = previous.get(name, {})
            row.update(
                status="cached_verified",
                sha256=sha,
                bytes=p.stat().st_size,
                retrieved_utc=prior.get("retrieved_utc"),
                dryad_sha256_match=True,
            )
            return row
        if args.offline:
            raise FileNotFoundError(p)
        row["attempts"] = []
        for u in [public, mirror]:
            try:
                data = get(u)
                sha = hashlib.sha256(data).hexdigest()
                if sha != expected:
                    raise ValueError(
                        "Downloaded bytes do not match published Dryad SHA256"
                    )
                p.write_bytes(data)
                row.update(
                    status="downloaded_verified",
                    url=u,
                    sha256=sha,
                    bytes=len(data),
                    retrieved_utc=datetime.datetime.now(
                        datetime.timezone.utc
                    ).isoformat(),
                    dryad_sha256_match=True,
                )
                return row
            except Exception as e:
                row["attempts"].append({"url": u, "error": str(e)})
        row["status"] = "failed"
        return row

    rows = list(ThreadPoolExecutor(max_workers=4).map(acquire, FILES.items()))
    for name, u in META.items():
        p = R / name
        if not p.exists():
            if args.offline:
                raise FileNotFoundError(p)
            p.write_bytes(get(u))
        rows.append(
            {
                "name": name,
                "path": str(p.relative_to(B)),
                "url": u,
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                "bytes": p.stat().st_size,
                "status": "metadata_cached",
                "license": "public repository metadata",
            }
        )
    old.write_text(json.dumps(rows, indent=2) + "\n")
    for r in rows:
        print(r["name"], r["status"])
    raise SystemExit(any(r["status"] == "failed" for r in rows))


if __name__ == "__main__":
    main()
