#!/usr/bin/env python3
"""Burn a print manifest's colours into the STEP, so the CAD and the print agree.

Usage::

    python -m hybrid_detector.cli.paint_step \
        assets/cad/rig0907.step \
        --manifest assets/print/rig0907_patterns_RGB_exact1mm_manifest.json \
        --out assets/cad/rig0907_rgb.step \
        --drop-face 1345

Why this exists.  The Onshape export paints the facets in whatever colours the
CAD author was working in, and the printed decals are a later, separate
decision -- on this rig the print is what makes the skin chiral, and the export
is not.  Feeding the raw export to ``build_hybrid_carrier_model`` therefore
needs a stack of ``--repaint`` flags, and *forgetting them is silent*: the
descriptor still builds, and it is simply not chiral.  A flag you must remember
is a defect.  This tool moves the print's colours into the file the builder
reads, so the builder's defaults become correct and the next export can be
repainted the same way in one command.

The manifest keys faces by their **position in ``ADVANCED_FACE`` file order**,
which is what a human reading the pattern sheet has; the STEP keys them by
entity id, which is what survives a re-export intact.  The two are reconciled
here, once, and the mapping is printed so it can be checked by eye.

Colours are written as ``COLOUR_RGB`` in the exact ink values the sheet prints,
not as named CAD colours, because ``semantic_colour`` reads RGB and only a
handful of names.  A face named by ``--drop-face`` loses its paint entirely:
that is how a decal that turned out to be unprintable is recorded in the CAD
rather than in someone's memory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from hybrid_detector.step import _split_statements, read_step, semantic_colour, split_arguments

# Everything the appearance chain is made of.  Entities of these kinds are
# rebuilt and garbage-collected; nothing else in the file is touched.
PRESENTATION_KEYWORDS = frozenset(
    {
        "OVER_RIDING_STYLED_ITEM",
        "PRESENTATION_STYLE_ASSIGNMENT",
        "SURFACE_STYLE_USAGE",
        "SURFACE_SIDE_STYLE",
        "SURFACE_STYLE_FILL_AREA",
        "FILL_AREA_STYLE",
        "FILL_AREA_STYLE_COLOUR",
        "COLOUR_RGB",
        "DRAUGHTING_PRE_DEFINED_COLOUR",
    }
)


class Statement:
    __slots__ = ("entity_id", "keyword", "body")

    def __init__(self, entity_id: int, keyword: str, body: str) -> None:
        self.entity_id = entity_id
        self.keyword = keyword
        self.body = body

    def text(self) -> str:
        if self.keyword == "_COMPLEX":
            return f"#{self.entity_id}={self.body};"
        return f"#{self.entity_id}={self.keyword}({self.body});"

    def refs(self) -> set[int]:
        return {int(x) for x in re.findall(r"#(\d+)", self.body)}


def parse(text: str) -> tuple[str, list[Statement], str]:
    head, _, rest = text.partition("DATA;")
    data_text, _, tail = rest.rpartition("ENDSEC;")
    statements: list[Statement] = []
    for raw in _split_statements(data_text):
        raw = raw.strip()
        if not raw.startswith("#"):
            continue
        prefix, _, body_text = raw.partition("=")
        entity_id = int(prefix.strip()[1:])
        body = re.sub(r"\s+", "", body_text.strip())
        match = re.match(r"^([A-Z_0-9]+)\((.*)\)$", body, flags=re.S)
        if match is None:
            statements.append(Statement(entity_id, "_COMPLEX", body))
        else:
            statements.append(Statement(entity_id, match.group(1), match.group(2)))
    return head + "DATA;", statements, "ENDSEC;" + tail


def hex_to_rgb(value: str) -> tuple[float, float, float]:
    value = value.strip().lstrip("#")
    if len(value) != 6:
        raise SystemExit(f"colour {value!r} is not a 6-digit hex triple")
    return tuple(int(value[i : i + 2], 16) / 255.0 for i in (0, 2, 4))  # type: ignore[return-value]


def colour_literal(rgb: tuple[float, float, float]) -> str:
    return "'',{}".format(",".join(repr(round(float(c), 15)) for c in rgb))


def face_order(statements: list[Statement]) -> list[int]:
    """Entity ids of ADVANCED_FACE in file order -- the manifest's index space."""
    return [s.entity_id for s in statements if s.keyword == "ADVANCED_FACE"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("step", type=Path, help="the raw CAD export to repaint")
    parser.add_argument(
        "--manifest", type=Path, required=True, help="the print manifest that owns the colours"
    )
    parser.add_argument("--out", type=Path, required=True, help="where to write the repainted STEP")
    parser.add_argument(
        "--drop-face",
        type=int,
        action="append",
        default=[],
        metavar="ENTITY",
        help=(
            "also strip this STEP face's paint, on top of the manifest's not_applied list. The CAD "
            "then says the decal is not applied, and the builder sees a bare face. Repeatable."
        ),
    )
    parser.add_argument(
        "--body-colour",
        default=None,
        help="also repaint the unpainted body, as #RRGGBB. Default: leave the export's body colour alone.",
    )
    args = parser.parse_args(argv)

    manifest = json.loads(args.manifest.read_text())
    inks = {k: hex_to_rgb(v) for k, v in manifest["colors"].items()}
    wanted_by_index = {int(k): v for k, v in manifest["color_faces"].items()}
    # A decal that is printed but not applied is a property of the print, so it
    # lives in the manifest.  A flag you have to remember is the same defect
    # this tool exists to remove.
    not_applied = [int(i) for i in manifest.get("not_applied", [])]

    text = args.step.read_text()
    head, statements, tail = parse(text)
    by_id = {s.entity_id: s for s in statements}
    faces = face_order(statements)

    # -- index space -> entity space, checked, not assumed ------------------
    missing = sorted(i for i in set(wanted_by_index) | set(not_applied) if i >= len(faces))
    if missing:
        raise SystemExit(
            f"manifest names face index {missing}, but the STEP has only {len(faces)} faces"
        )
    wanted: dict[int, str] = {faces[i]: wanted_by_index[i] for i in wanted_by_index}
    dropped = {faces[i] for i in not_applied if i < len(faces)} | {int(x) for x in args.drop_face}
    unknown = sorted(dropped - set(by_id))
    if unknown:
        raise SystemExit(f"--drop-face names {unknown}, which are not entities in this STEP")
    not_faces = sorted(d for d in dropped if by_id[d].keyword != "ADVANCED_FACE")
    if not_faces:
        raise SystemExit(f"--drop-face names {not_faces}, which are not ADVANCED_FACE")

    painted_now = {
        int(split_arguments(s.body)[2].lstrip("#"))
        for s in statements
        if s.keyword == "OVER_RIDING_STYLED_ITEM"
    }
    stray = sorted(painted_now - set(wanted) - dropped)
    if stray:
        raise SystemExit(
            f"the STEP paints face(s) {stray} that the manifest does not print. Either add them to the "
            "manifest or pass --drop-face for each; leaving CAD paint the print does not have is "
            "how a descriptor ends up describing an object nobody built."
        )

    keep = {f: c for f, c in wanted.items() if f not in dropped}

    # -- rebuild the appearance chain --------------------------------------
    # Every painted face gets its own fresh chain.  Rewriting the existing ones
    # in place would be shorter and wrong the day two faces share a chain: the
    # second repaint would silently move the first face's colour too.
    survivors = [s for s in statements if s.keyword != "OVER_RIDING_STYLED_ITEM"]
    body_styled = next((s for s in survivors if s.keyword == "STYLED_ITEM"), None)
    if body_styled is None:
        raise SystemExit("no STYLED_ITEM in this STEP; nothing to hang face overrides off")

    next_id = max(s.entity_id for s in statements) + 1

    def add(keyword: str, body: str) -> int:
        nonlocal next_id
        statement = Statement(next_id, keyword, body)
        survivors.append(statement)
        next_id += 1
        return statement.entity_id

    colour_id: dict[str, int] = {}
    for key in sorted(set(keep.values())):
        if key not in inks:
            raise SystemExit(f"manifest paints faces {key!r} but defines no such colour")
        colour_id[key] = add("COLOUR_RGB", colour_literal(inks[key]))

    styled_item_ids: list[int] = []
    for face in sorted(keep):
        fasc = add("FILL_AREA_STYLE_COLOUR", f"'',#{colour_id[keep[face]]}")
        fas = add("FILL_AREA_STYLE", f"'',(#{fasc})")
        ssfa = add("SURFACE_STYLE_FILL_AREA", f"#{fas}")
        sss = add("SURFACE_SIDE_STYLE", f"'',(#{ssfa})")
        ssu = add("SURFACE_STYLE_USAGE", f".BOTH.,#{sss}")
        psa = add("PRESENTATION_STYLE_ASSIGNMENT", f"(#{ssu})")
        styled_item_ids.append(
            add("OVER_RIDING_STYLED_ITEM", f"'',(#{psa}),#{face},#{body_styled.entity_id}")
        )

    if args.body_colour is not None:
        chain = [body_styled.entity_id]
        seen: set[int] = set()
        target = None
        while chain:
            current = chain.pop()
            if current in seen:
                continue
            seen.add(current)
            statement = next((s for s in survivors if s.entity_id == current), None)
            if statement is None:
                continue
            if statement.keyword == "FILL_AREA_STYLE_COLOUR":
                target = statement
                break
            chain.extend(sorted(statement.refs()))
        if target is None:
            raise SystemExit("the body STYLED_ITEM has no FILL_AREA_STYLE_COLOUR to repaint")
        target.body = f"'',#{add('COLOUR_RGB', colour_literal(hex_to_rgb(args.body_colour)))}"

    # The presentation representation lists the styled items by hand.
    for statement in survivors:
        if statement.keyword.endswith("PRESENTATION_REPRESENTATION"):
            argv_ = split_arguments(statement.body)
            items = [f"#{body_styled.entity_id}"] + [f"#{i}" for i in styled_item_ids]
            argv_[1] = "(" + ",".join(items) + ")"
            statement.body = ",".join(argv_)

    # -- sweep the chain entities nothing points at any more ----------------
    swept: list[int] = []
    while True:
        referenced: set[int] = set()
        for statement in survivors:
            referenced |= statement.refs()
        orphans = [
            s
            for s in survivors
            if s.keyword in PRESENTATION_KEYWORDS
            and s.entity_id not in referenced
            and s.keyword != "OVER_RIDING_STYLED_ITEM"
        ]
        if not orphans:
            break
        swept.extend(s.entity_id for s in orphans)
        gone = {s.entity_id for s in orphans}
        survivors = [s for s in survivors if s.entity_id not in gone]

    survivors.sort(key=lambda s: s.entity_id)

    # -- referential integrity, before anything is written ------------------
    ids = {s.entity_id for s in survivors}
    dangling = sorted({r for s in survivors for r in s.refs()} - ids)
    if dangling:
        raise SystemExit(
            f"internal error: repainted file would reference missing entities {dangling}"
        )

    # The note goes above HEADER, where the exporter puts its own: a comment
    # inside DATA is split on every ";" and every "\'" by any STEP reader that
    # tokenises the section, this repository included.
    note = (
        f"/* Repainted by hybrid_detector.cli.paint_step from {args.manifest.name}.\n"
        " * Face colours come from the print, not from CAD styling. Face ids and\n"
        f" * all geometry are the CAD export as-is. {len(keep)} faces painted, {len(dropped)} stripped.\n"
        f" * Source export {args.step.name} sha256 {hashlib.sha256(text.encode()).hexdigest()}\n"
        " * -- two exports of the same part on the same day are not the same file,\n"
        " * so the hash, not the name, says which one this was painted from. */\n"
    )
    marker = "ISO-10303-21;"
    if marker not in head:
        raise SystemExit("input does not start with an ISO-10303-21 header")
    head = head.replace(marker, marker + "\n" + note, 1)
    args.out.write_text(head + "\n" + "\n".join(s.text() for s in survivors) + "\n" + tail)

    # -- read it back the way the builder will ------------------------------
    model = read_step(args.out)
    got = {int(f.entity_id): semantic_colour(f.colour) for f in model.faces if f.colour_is_override}
    expected_names = {f: semantic_colour(inks[c]) for f, c in keep.items()}
    if got != expected_names:
        wrong = {
            f: (got.get(f), expected_names.get(f))
            for f in set(got) | set(expected_names)
            if got.get(f) != expected_names.get(f)
        }
        raise SystemExit(f"read-back mismatch face -> (got, wanted): {wrong}")

    index_of = {entity: i for i, entity in enumerate(faces)}
    print(f"[OK] {args.step.name} -> {args.out.name}")
    for face in sorted(keep):
        rgb = inks[keep[face]]
        print(
            "  face #{eid} (manifest index {idx:>2}) {key} {hexes} -> {name}".format(
                eid=face,
                idx=index_of[face],
                key=keep[face],
                hexes=manifest["colors"][keep[face]],
                name=semantic_colour(rgb),
            )
        )
    for face in sorted(dropped):
        print(
            f"  face #{face} (manifest index {index_of.get(face, '?')}) paint stripped -- decal not applied"
        )
    print(f"[OK] {len(keep)} overrides written, {len(swept)} orphaned appearance entities swept")
    print(f"[OK] read back through step_brep: {len(got)} painted faces, all colours as printed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
