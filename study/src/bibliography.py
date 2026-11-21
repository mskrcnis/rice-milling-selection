import csv
import json
import re
from pathlib import Path
from .common import ROOT


def escape(text):
    value = (
        str(text)
        .replace("‐", "-")
        .replace("–", "-")
        .replace("—", "-")
        .replace("\u00a0", " ")
    )
    value = re.sub(r"\s+", " ", value).strip()
    for a, b in [
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
    ]:
        value = value.replace(a, b)
    return value


def create_references(destination):
    rows = list(csv.DictReader((ROOT / "audit/prior_art_closest_work.csv").open()))
    output = [r"\begin{thebibliography}{99}"]
    metadata = []
    for row in rows:
        paths = list((ROOT / "raw/literature").glob(row["study_id"] + "_crossref.json"))
        message = json.loads(paths[0].read_text()).get("message", {}) if paths else {}
        journal = (message.get("container-title") or [""])[0]
        year = {
            "rebollo2023": 2023,
            "scheffel2025": 2025,
            "monteverde2018": 2018,
            "tanaka2018": 2018,
            "tsai2021": 2021,
            "tu2024": 2026,
            "chen2024": 2024,
            "liu2026": 2026,
            "dhakal2025": 2025,
            "clarke2024": 2024,
            "active2021": 2021,
        }[row["study_id"]]
        authors = row["authors"].split("; ")
        author = (authors[0] + " et al.") if len(authors) > 3 else "; ".join(authors)
        details = ", ".join(
            str(x) for x in [journal, message.get("volume"), message.get("page")] if x
        )
        entry = f"{escape(author)} ({year}). {escape(row['title'])}. {escape(details)}."
        entry = entry.replace(". .", ".")
        output.append(
            r"\bibitem{"
            + row["study_id"]
            + "} "
            + entry
            + r" \href{https://doi.org/"
            + row["doi"]
            + "}{doi:"
            + row["doi"]
            + "}."
        )
        if row["study_id"] == "tu2024":
            output[-1] += " First published online 2024; issue publication 2026."
        metadata.append(
            {
                "key": row["study_id"],
                "doi": row["doi"],
                "year_used": year,
                "verification": row["verification_notes"],
            }
        )
    output.append(r"\end{thebibliography}")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "references.tex").write_text("\n\n".join(output) + "\n")
    (destination / "reference_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )


if __name__ == "__main__":
    create_references(ROOT / "manuscript")
