#!/usr/bin/env python3
"""Turn the Hybrid Carrier STEP export into the descriptor the detector reads.

Usage::

    python -m hybrid_detector.cli.build_model finalrig0904.step \
        --out assets/models/hybrid_carrier_v1_20260904.json \
        --anchor-ids 30 31

What the STEP says, and what it does not:

* an unpainted 60 x 60 mm square face is an ArUco pad (the roadmap's two ID
  anchors) -- the CAD author leaves them unpainted precisely so a sticker can
  go there, so "no override colour" is the signal, not a naming convention;
* every ``OVER_RIDING_STYLED_ITEM`` face is a painted facet;
* nothing in the file records the marker ids, the printed black stroke width,
  or how that stroke is aligned to the CAD edge.  Those are print decisions and
  arrive as flags, recorded in the output so a later reader can see they were
  assumptions rather than measurements.

The output also carries the *derived* things the detector cannot rediscover per
frame: which faces share which edge, what the printed ink does at each of those
edges, and a triangle soup of the whole planar skin for self-occlusion tests.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from hybrid_detector.step import StepFace, StepModel, read_step, semantic_colour

PAINT_COLOURS = {"red", "blue", "black", "green", "yellow", "magenta", "cyan", "grey", "white"}
BODY = "body"


def _round(value: Any, digits: int = 6) -> Any:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        return round(float(array), digits)
    return np.round(array, digits).tolist()


# ---------------------------------------------------------------------------
# face roles
# ---------------------------------------------------------------------------


def is_anchor_pad(face: StepFace, pad_size_mm: float, tolerance_mm: float) -> bool:
    if not face.is_plane or face.colour_is_override or face.outer_polygon is None:
        return False
    polygon = face.outer_polygon
    if len(polygon) != 4 or face.inner_polygons:
        return False
    lengths = [float(np.linalg.norm(polygon[(i + 1) % 4] - polygon[i])) for i in range(4)]
    if max(abs(length - pad_size_mm) for length in lengths) > tolerance_mm:
        return False
    return abs(float(face.area or 0.0) - pad_size_mm**2) <= 4.0 * pad_size_mm * tolerance_mm


def _is_pad_shaped(face: StepFace, pad_size_mm: float, tolerance_mm: float) -> bool:
    """Square of the right size, whatever the CAD painted it."""
    polygon = face.outer_polygon
    if not face.is_plane or polygon is None or len(polygon) != 4 or face.inner_polygons:
        return False
    lengths = [float(np.linalg.norm(polygon[(i + 1) % 4] - polygon[i])) for i in range(4)]
    return max(abs(length - pad_size_mm) for length in lengths) <= tolerance_mm


def _back_to_back(pads: Sequence[StepFace], gap_mm: float = 6.0) -> set[int]:
    """The pads that are the far side of a tab whose near side is also a pad.

    A tab has two faces and both can take a sticker.  Naming them apart matters
    because they sit on the same square of material and differ only in which way
    they look, so the side of the symmetry plane -- what tells the front pads
    apart -- says nothing here.  The one facing along +Z is called the front.
    """
    reverse: set[int] = set()
    for a, b in itertools.combinations(pads, 2):
        if float(np.dot(a.normal, b.normal)) > -0.99:
            continue
        if float(np.linalg.norm(a.centroid - b.centroid)) > gap_mm:
            continue
        back = a if float(a.normal[2]) < float(b.normal[2]) else b
        reverse.add(int(back.entity_id))
    return reverse


def direction_name(normal: np.ndarray) -> str:
    """A short, frame-explicit label; no invented left/right convention."""
    axis = int(np.argmax(np.abs(normal)))
    sign = "p" if normal[axis] >= 0 else "n"
    return f"{sign}{'xyz'[axis]}"


def side_name(centroid: np.ndarray, axis: int = 1, deadband: float = 5.0) -> str:
    """Which side of the rig's symmetry plane a face sits on."""
    value = float(centroid[axis])
    letter = "xyz"[axis]
    if abs(value) <= deadband:
        return f"{letter}c"
    return f"{letter}p" if value > 0 else f"{letter}n"


def facet_name(face: StepFace, colour: str, used: set[str]) -> str:
    """``colour_normal`` where that is unique, plus the side when it is not.

    Mirror twins differ only by which side of ``y = 0`` they sit on, so that is
    the disambiguator worth spending a suffix on -- an anonymous ``_2`` would
    hide the one distinction the rest of this file keeps warning about.
    """
    base = f"{colour}_{direction_name(face.normal)}"
    if base not in used:
        used.add(base)
        return base
    name = f"{base}_{side_name(face.centroid)}"
    suffix = 2
    while name in used:
        name = f"{base}_{side_name(face.centroid)}_{suffix}"
        suffix += 1
    used.add(name)
    return name


def back_face_of(face: StepFace, pads: list[StepFace], max_gap_mm: float = 8.0) -> StepFace | None:
    """The anchor pad this painted face is the reverse side of, if any.

    A face that is parallel to a pad, faces the other way, and sits a plate
    thickness behind it is the inside of that plate.  It is real geometry and it
    can even be seen from behind, but it is not a facet of the outward skin the
    detector reasons about, and its area is the ArUco pad's area, so leaving it
    in the facet list would have the model expect paint where the sticker is.
    """
    for pad in pads:
        if float(np.dot(face.normal, pad.normal)) > -0.99:
            continue
        gap = float(np.dot(pad.centroid - face.centroid, pad.normal))
        if 0.0 < gap <= max_gap_mm:
            offset = (face.centroid - pad.centroid) - gap * -pad.normal
            if float(np.linalg.norm(offset)) < 5.0:
                return pad
    return None


# ---------------------------------------------------------------------------
# what the printed black stroke does at an edge
# ---------------------------------------------------------------------------


def stroke_interval(width: float, alignment: str, side: str) -> tuple[float, float]:
    """The stroke a painted face draws, signed along the A -> B edge normal."""
    table = {
        "inside": (-width, 0.0),
        "centred": (-0.5 * width, 0.5 * width),
        "outside": (0.0, width),
    }
    low, high = table[alignment]
    return (low, high) if side == "a" else (-high, -low)


def dark_interval(
    colour_a: str,
    colour_b: str,
    width: float,
    alignment: str,
    *,
    carries_ink_a: bool = True,
    carries_ink_b: bool = True,
) -> tuple[bool, float | None, float | None]:
    """Bounds of the ink straddling an edge; ``None`` means it runs off that side.

    A black *fill* is unbounded on its own side, which is exactly why a black
    facet is a poor metric neighbour: the boundary you can see is one stroke
    width away from the CAD edge instead of centred on it.

    A stroke specified as centred straddles the edge only if both surfaces can
    take ink.  An ArUco pad cannot -- the sticker is on top of it -- so a facet's
    stroke there stops at the edge and the band it leaves is one-sided, half a
    stroke off centre.  Modelling that as centred would bake a fixed error into
    every pose the anchors contribute to.
    """
    intervals: list[tuple[float, float]] = []
    if colour_a == "black":
        intervals.append((-math.inf, 0.0))
    if colour_b == "black":
        intervals.append((0.0, math.inf))

    def clip(interval: tuple[float, float]) -> tuple[float, float] | None:
        low, high = interval
        if not carries_ink_a:
            low = max(low, 0.0)
        if not carries_ink_b:
            high = min(high, 0.0)
        return (low, high) if high > low else None

    for colour, side, carries in ((colour_a, "a", carries_ink_a), (colour_b, "b", carries_ink_b)):
        if colour in (BODY, "white") or not carries:
            continue
        clipped = clip(stroke_interval(width, alignment, side))
        if clipped is not None:
            intervals.append(clipped)
    if not intervals:
        return False, None, None
    low = min(i[0] for i in intervals)
    high = max(i[1] for i in intervals)
    return True, (None if math.isinf(low) else low), (None if math.isinf(high) else high)


def pivot_socket(step: StepModel) -> dict[str, Any]:
    """The seating socket, read off the STEP rather than asserted in prose.

    The part studio puts the socket sphere centre on the origin so that the rig
    frame *is* the frame of the point the rig can be independently localised at
    -- a pivot fit therefore has a known answer, ``p_rig = 0``, and any
    departure from it reads out the error of the whole chain instead of needing
    a separate alignment to interpret.  That is a load-bearing property, so the
    builder measures it and says how far off it is rather than reprinting the
    claim from the previous revision.

    ``marker_rig_20260818_cad.json`` records the same block for the previous
    revision of this part studio, which is why the field names match it.
    """
    spheres = [face for face in step.faces if face.surface_type == "sphere"]
    if not spheres:
        return {
            "socket_sphere_centre_mm": None,
            "socket_sphere_radius_mm": None,
            "note": "no spherical face in this STEP; the rig frame has no seating socket to sit on",
        }
    # The rig carries small blend spheres as well as the socket; the socket is
    # the big one, and picking it by radius is what the 0818 extraction did.
    socket = max(spheres, key=lambda face: float(face.surface_params["radius"]))
    centre = np.asarray(socket.surface_params["origin"], dtype=np.float64)
    return {
        "socket_sphere_centre_mm": _round(centre.tolist()),
        "socket_sphere_radius_mm": _round(float(socket.surface_params["radius"])),
        "socket_face_entity_id": int(socket.entity_id),
        "offset_from_origin_mm": _round(float(np.linalg.norm(centre))),
        "note": (
            "the socket centre is the rig's TCP: it is the one point on the rig whose "
            "position is fixed by a seated ball rather than by the paint"
        ),
    }


def triangulate(polygon: np.ndarray) -> list[list[list[float]]]:
    return [
        [polygon[0].tolist(), polygon[i].tolist(), polygon[i + 1].tolist()]
        for i in range(1, len(polygon) - 1)
    ]


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------


def build_model(
    step: StepModel,
    *,
    anchor_ids: list[int],
    dictionary: str,
    pad_size_mm: float,
    marker_size_mm: float,
    pad_tolerance_mm: float,
    pad_faces: Sequence[int] | None = None,
    anchor_id_by_face: dict[int, int] | None = None,
    border_width_mm: float,
    border_alignment: str,
    sticker_covers_pad: bool,
    carrier_id: str,
    repaint: dict[str, str] | None = None,
    paste_quadrant_by_id: dict[int, int] | None = None,
) -> dict[str, Any]:
    forced = {int(entity) for entity in (pad_faces or ())}
    pads = [
        f
        for f in step.faces
        if is_anchor_pad(f, pad_size_mm, pad_tolerance_mm) or int(f.entity_id) in forced
    ]
    missing = forced - {int(f.entity_id) for f in pads}
    if missing:
        raise SystemExit(f"--pad-face named {sorted(missing)}, which the STEP does not have")
    for face in pads:
        if int(face.entity_id) in forced and not _is_pad_shaped(
            face, pad_size_mm, pad_tolerance_mm
        ):
            raise SystemExit(
                f"--pad-face {face.entity_id} is not a {pad_size_mm:g} mm square; a sticker needs "
                "a face the size of the sticker"
            )
    # Sort by +Y then +Z then +X so re-running on a revised STEP keeps ids stable.
    pads.sort(
        key=lambda f: (round(f.centroid[1], 3), round(f.centroid[2], 3), round(f.centroid[0], 3))
    )
    explicit = {int(k): int(v) for k, v in (anchor_id_by_face or {}).items()}
    unknown_ids = explicit.keys() - {int(f.entity_id) for f in pads}
    if unknown_ids:
        raise SystemExit(f"--anchor-id named face(s) {sorted(unknown_ids)}, which are not pads")
    free = [i for i in anchor_ids if i not in set(explicit.values())]
    if len(pads) != len(explicit) + len(free):
        raise SystemExit(
            f"found {len(pads)} {pad_size_mm:g} mm square pads but got "
            f"{len(explicit) + len(free)} ids; pass one id per pad"
        )
    assigned = [explicit.get(int(pad.entity_id)) for pad in pads]
    spare = iter(free)
    assigned = [next(spare) if value is None else value for value in assigned]

    reverse_of = _back_to_back(pads)
    quadrants = {int(k): int(v) for k, v in (paste_quadrant_by_id or {}).items()}
    unknown_quadrant_ids = quadrants.keys() - set(assigned)
    if unknown_quadrant_ids:
        raise SystemExit(
            f"--paste-quadrant named marker id(s) {sorted(unknown_quadrant_ids)}, which no pad carries"
        )
    if any(value not in (0, 1, 2, 3) for value in quadrants.values()):
        raise SystemExit(
            "--paste-quadrant takes 0, 1, 2 or 3 (90 deg steps of the sticker on its pad)"
        )
    anchors: list[dict[str, Any]] = []
    pad_by_name: dict[str, StepFace] = {}
    for marker_id, pad in zip(assigned, pads, strict=True):
        polygon = pad.outer_polygon
        centre = polygon.mean(axis=0)
        # ArUco reports corners clockwise about the outward normal, so the object
        # points must be too; the CAD loop is wound counter-clockwise.
        aruco_order = np.vstack([polygon[:1], polygon[:0:-1]])
        marker = centre + (marker_size_mm / pad_size_mm) * (aruco_order - centre)
        # Both pads share a 45 deg normal, so the dominant-axis label ties; the
        # side of the symmetry plane is what actually tells them apart.
        name = f"anchor_{side_name(centre)}" + ("_rev" if int(pad.entity_id) in reverse_of else "")
        pad_by_name[name] = pad
        anchors.append(
            {
                "name": name,
                "marker_id": int(marker_id),
                "dictionary": dictionary,
                "face_entity_id": pad.entity_id,
                "pad_corners": _round(polygon),
                "marker_corners": _round(marker),
                "corner_order": "ArUco order: clockwise about the outward normal",
                "normal": _round(pad.normal),
                "centre": _round(centre),
                "pad_size": pad_size_mm,
                "marker_size": marker_size_mm,
                "quiet_zone": round(0.5 * (pad_size_mm - marker_size_mm), 6),
                "paste_quadrant": quadrants.get(int(marker_id)),
                "paste_quadrant_note": (
                    "physical sticker rotation, measured on the built rig"
                    if int(marker_id) in quadrants
                    else "physical sticker rotation, unknown until assembly; the detector "
                    "enumerates 0/90/180/270 deg until this is measured and written here"
                ),
            }
        )

    facets: list[dict[str, Any]] = []
    face_by_name: dict[str, StepFace] = dict(pad_by_name)
    colour_by_name: dict[str, str] = dict.fromkeys(pad_by_name, "white")
    used_names = set(pad_by_name)

    # --repaint is keyed by the name the face gets from the colour the STEP
    # carries, because that is the only name a reader has before deciding to
    # repaint.  So resolve those names first, then let the new colour rename
    # the facet: a facet called black_ny that gets painted green should not go
    # on calling itself black.
    painted = [
        f for f in step.faces if f.colour_is_override and f.is_plane and f.outer_polygon is not None
    ]
    step_names = set(pad_by_name)
    name_in_step = {
        face.entity_id: facet_name(face, semantic_colour(face.colour) or "unknown", step_names)
        for face in painted
    }
    repaint = {str(k): str(v) for k, v in (repaint or {}).items()}
    unknown = sorted(set(repaint) - set(name_in_step.values()))
    if unknown:
        raise SystemExit(
            f"--repaint names {', '.join(unknown)}, which the STEP does not have. "
            f"Its painted faces are: {', '.join(sorted(name_in_step.values()))}"
        )
    bad = sorted({c for c in repaint.values() if c not in PAINT_COLOURS})
    if bad:
        raise SystemExit(f"--repaint colour(s) {', '.join(bad)} not in {sorted(PAINT_COLOURS)}")

    for face in painted:
        step_colour = semantic_colour(face.colour) or "unknown"
        colour = repaint.get(name_in_step[face.entity_id], step_colour)
        name = facet_name(face, colour, used_names)
        pad = back_face_of(face, pads)
        note = ""
        use_for_pose = True
        if pad is not None:
            use_for_pose = False
            note = (
                f"reverse side of ArUco pad #{pad.entity_id}; only visible from behind the rig "
                "and its mirror twin is unpainted, so it reads as a stray click rather than a feature"
            )
        facets.append(
            {
                "name": name,
                "colour": colour,
                "face_entity_id": face.entity_id,
                "polygon": _round(face.outer_polygon),
                "winding": "counter-clockwise about the outward normal",
                "normal": _round(face.normal),
                "centroid": _round(face.centroid),
                "area": _round(face.area or 0.0),
                "num_vertices": int(len(face.outer_polygon)),
                "use_for_pose": use_for_pose,
                "note": note,
                "step_colour": step_colour,
                "repainted": colour != step_colour,
            }
        )
        face_by_name[name] = face
        colour_by_name[name] = colour

    name_by_entity = {face.entity_id: name for name, face in face_by_name.items()}
    pose_names = set(pad_by_name) | {f["name"] for f in facets if f["use_for_pose"]}
    planar = [f for f in step.faces if f.is_plane and f.outer_polygon is not None]

    edges: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for name, face in face_by_name.items():
        polygon = face.outer_polygon
        for k, edge_id in enumerate(face.outer_edge_ids):
            neighbours = [
                other
                for other in planar
                if other.entity_id != face.entity_id and edge_id in other.edge_ids
            ]
            if len(neighbours) != 1:
                continue
            other = neighbours[0]
            key = (min(face.entity_id, other.entity_id), max(face.entity_id, other.entity_id))
            if (key, edge_id) in seen:
                continue
            seen.add((key, edge_id))
            other_name = name_by_entity.get(other.entity_id, f"body_{other.entity_id}")
            colour_a = colour_by_name.get(name, BODY)
            colour_b = colour_by_name.get(other_name, BODY)
            p0, p1 = polygon[k], polygon[(k + 1) % len(polygon)]
            has_ink, low, high = dark_interval(
                colour_a,
                colour_b,
                border_width_mm,
                border_alignment,
                carries_ink_a=not (sticker_covers_pad and name in pad_by_name),
                carries_ink_b=not (sticker_covers_pad and other_name in pad_by_name),
            )
            cosine = float(np.clip(np.dot(face.normal, other.normal), -1.0, 1.0))
            # Convex if stepping across the edge turns the surface away from the
            # material, which is what makes the edge a silhouette from some views.
            midpoint = 0.5 * (p0 + p1)
            convex = float(np.dot(other.centroid - midpoint, face.normal)) < 0.0
            edges.append(
                {
                    "face_a": name,
                    "face_b": other_name,
                    "colour_a": colour_a,
                    "colour_b": colour_b,
                    "step_edge_id": int(edge_id),
                    "p0": _round(p0),
                    "p1": _round(p1),
                    "ref_b": _round(other.centroid),
                    "normal_b": _round(other.normal),
                    "length": _round(float(np.linalg.norm(p1 - p0))),
                    "dihedral_deg": _round(180.0 - math.degrees(math.acos(cosine))),
                    "convex": bool(convex),
                    "use_for_pose": bool(name in pose_names or other_name in pose_names),
                    "has_ink": bool(has_ink),
                    "dark_lo": None if low is None else _round(low),
                    "dark_hi": None if high is None else _round(high),
                    "profile": (
                        "unmeasurable"
                        if not has_ink or (low is None and high is None)
                        else "band_centre"
                        if (low is not None and high is not None)
                        else ("step_dark_positive" if high is None else "step_dark_negative")
                    ),
                }
            )

    triangles: list[list[list[float]]] = []
    triangle_faces: list[int] = []
    for face in planar:
        fan = triangulate(face.outer_polygon)
        triangles.extend(fan)
        triangle_faces.extend([face.entity_id] * len(fan))

    return {
        "schema": "hybrid_carrier_cad/v1",
        "carrier_id": carrier_id,
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "units": "mm",
        "frame": (
            "STEP part-studio frame, not recentred; the pivot socket sphere centre sits on "
            "the origin, so a pivot fit on a track built from this model must return p_rig = 0"
        ),
        "pivot": pivot_socket(step),
        "source": {
            "step_file": step.path.name,
            "sha256": step.sha256,
            "part_name": step.header.get("FILE_NAME", [""])[0],
            "step_timestamp": step.header.get("FILE_NAME", ["", ""])[1]
            if len(step.header.get("FILE_NAME", [])) > 1
            else "",
            "originating_system": step.header.get("FILE_NAME", [""] * 7)[6]
            if len(step.header.get("FILE_NAME", [])) > 6
            else "",
            "native_units": step.units,
            "body_colour_rgb": list(step.body_colour)
            if isinstance(step.body_colour, tuple)
            else step.body_colour,
            "num_faces": len(step.faces),
            "reader": "hybrid_detector.step (STEP entity ids kept attached to geometry)",
        },
        "print": {
            "body": "unpainted faces stay bare; they are the light reference the facets sit against",
            "facet_fill": "solid paint, one colour per facet",
            "repaint": dict(repaint),
            "repaint_note": (
                "colours the print carries that the STEP does not; each facet keeps its STEP "
                "colour under step_colour so the CAD and the print can be diffed"
            ),
            "border": {
                "width_mm": border_width_mm,
                "alignment": border_alignment,
                "assumed": True,
                "note": (
                    "not recorded in the STEP. Where two painted facets meet, the two strokes "
                    "merge into one band whose centre is the CAD edge for any alignment, so those "
                    "edges stay unbiased even if this number is wrong; elsewhere the value sets "
                    "the offset the detector removes, and detect_hybrid_carrier reports the band "
                    "width it actually measured so this can be checked against the print"
                ),
            },
            "anchor": {
                "layout": "white ArUco sticker centred on each unpainted pad, quiet zone left white",
                "sticker_covers_pad": sticker_covers_pad,
                "note": (
                    "False means the sticker is marker-sized and the rest of the 60 mm pad stays bare, "
                    "so a facet's stroke can wrap onto the pad rim and the pad-to-facet edges stay "
                    "self-centring. True means the sticker covers the whole pad, the stroke stops at "
                    "the edge, and those edges pick up half a stroke of bias. The stroke lands about "
                    "6 mm outside the marker either way, so it never touches the quiet zone"
                ),
            },
        },
        "anchors": anchors,
        "facets": facets,
        "edges": edges,
        "occluders": {
            "triangles": [[_round(v) for v in tri] for tri in triangles],
            "face_entity_ids": triangle_faces,
            "note": (
                "every planar face, fan-triangulated over its outer loop; inner loops (the two "
                "15 mm bores) are filled, which can only over-report occlusion, never under-report"
            ),
        },
    }


# ---------------------------------------------------------------------------
# what the geometry implies for detection
# ---------------------------------------------------------------------------


def mirror_twins(model: dict[str, Any], axis: int = 1, tolerance: float = 0.5) -> list[list[str]]:
    """Facets whose *geometry* maps onto each other under a reflection through ``axis`` = 0.

    Colour is deliberately ignored here.  The shape is what creates the
    ambiguity; the paint is what can resolve it, and separating the two is the
    only way to say how much of the skin actually carries chirality.
    """
    entries = [(f["name"], np.asarray(f["centroid"], dtype=np.float64)) for f in model["facets"]]
    twins: list[list[str]] = []
    taken: set[str] = set()
    for name, centroid in entries:
        if name in taken:
            continue
        mirrored = centroid.copy()
        mirrored[axis] *= -1.0
        for other_name, other_centroid in entries:
            if other_name == name or other_name in taken:
                continue
            if float(np.linalg.norm(other_centroid - mirrored)) < tolerance:
                twins.append([name, other_name])
                taken.update({name, other_name})
                break
    return twins


def chirality(model: dict[str, Any]) -> dict[str, Any]:
    """Which facets can say which side of y = 0 they are on, and which cannot.

    A facet whose mirror twin is painted the same colour is side-blind: seeing
    it tells you the shape but not the hand.  A facet whose twin is painted
    differently is a chirality carrier, and a view that contains one can be
    disambiguated without reading an anchor id.
    """
    colour_of = {f["name"]: f["colour"] for f in model["facets"]}
    usable = {f["name"] for f in model["facets"] if f["use_for_pose"]}
    area_of = {f["name"]: float(f["area"]) for f in model["facets"]}
    carriers: list[list[str]] = []
    blind: list[list[str]] = []
    for a, b in mirror_twins(model):
        if a not in usable or b not in usable:
            continue
        (carriers if colour_of[a] != colour_of[b] else blind).append([a, b])
    carrier_names = sorted({n for pair in carriers for n in pair})
    blind_names = sorted({n for pair in blind for n in pair})
    on_plane = sorted(
        f["name"]
        for f in model["facets"]
        if f["use_for_pose"]
        and abs(float(f["centroid"][1])) < 1.0
        and f["name"] not in carrier_names + blind_names
    )
    total_area = sum(area_of[n] for n in usable)
    return {
        "chirality_carrying_pairs": carriers,
        "side_blind_pairs": blind,
        "self_mirrored_facets": on_plane,
        "chirality_carrying_facets": carrier_names,
        "chirality_carrying_area_fraction": _round(
            sum(area_of[n] for n in carrier_names) / total_area if total_area else 0.0, 3
        ),
    }


def mirror_pairs(model: dict[str, Any], axis: int = 1, tolerance: float = 0.5) -> list[list[str]]:
    """Mirror twins that also share a colour -- the ones colour cannot tell apart."""
    colour_of = {f["name"]: f["colour"] for f in model["facets"]}
    pairs = [
        pair
        for pair in mirror_twins(model, axis, tolerance)
        if colour_of[pair[0]] == colour_of[pair[1]]
    ]
    return pairs


def alignment_comparison(model: dict[str, Any]) -> dict[str, float]:
    """How much unbiased edge each way of aligning the printed stroke buys.

    The stroke width is a guess until someone measures the print, but this
    comparison is not: it only depends on which side of the CAD edge the ink
    lands, and it is the difference between most of the rig's boundary being
    self-centring and almost none of it being so.
    """
    width = float(model["print"]["border"]["width_mm"])
    out: dict[str, float] = {}
    for alignment in ("inside", "centred", "outside"):
        total = 0.0
        covered = bool(model["print"]["anchor"]["sticker_covers_pad"])
        pads = {a["name"] for a in model["anchors"]} if covered else set()
        for edge in model["edges"]:
            if not edge["use_for_pose"]:
                continue
            has_ink, low, high = dark_interval(
                edge["colour_a"],
                edge["colour_b"],
                width,
                alignment,
                carries_ink_a=edge["face_a"] not in pads,
                carries_ink_b=edge["face_b"] not in pads,
            )
            if has_ink and low is not None and high is not None and abs(low + high) < 1e-9:
                total += float(edge["length"])
        out[alignment] = round(total, 2)
    return out


def diagnostics(model: dict[str, Any]) -> dict[str, Any]:
    facets = [f for f in model["facets"] if f["use_for_pose"]]
    colours: dict[str, int] = {}
    for facet in facets:
        colours[facet["colour"]] = colours.get(facet["colour"], 0) + 1

    same_colour = [
        [e["face_a"], e["face_b"], e["colour_a"]]
        for e in model["edges"]
        if e["colour_a"] == e["colour_b"] and e["colour_a"] not in (BODY, "white")
    ]

    measurable = [e for e in model["edges"] if e["profile"] != "unmeasurable" and e["use_for_pose"]]
    unbiased = [
        e
        for e in measurable
        if e["profile"] == "band_centre" and abs(e["dark_lo"] + e["dark_hi"]) < 1e-6
    ]

    points = np.vstack(
        [np.asarray(f["polygon"], dtype=np.float64) for f in facets]
        + [np.asarray(a["pad_corners"], dtype=np.float64) for a in model["anchors"]]
    )
    centroid = points.mean(axis=0)
    spread = float(np.sqrt(np.mean(np.sum((points - centroid) ** 2, axis=1))))

    normals = np.asarray(
        [f["normal"] for f in facets] + [a["normal"] for a in model["anchors"]], dtype=np.float64
    )
    gram = np.clip(normals @ normals.T, -1.0, 1.0)
    np.fill_diagonal(gram, -1.0)
    min_separation = float(np.degrees(np.arccos(np.max(gram))))

    return {
        "num_anchors": len(model["anchors"]),
        "num_facets_for_pose": len(facets),
        "num_facets_excluded": len(model["facets"]) - len(facets),
        "facet_colour_counts": colours,
        "facet_vertex_counts": sorted(f["num_vertices"] for f in facets),
        "same_colour_shared_edges": same_colour,
        "num_edges": len(model["edges"]),
        "num_measurable_edges": len(measurable),
        "num_unbiased_edges": len(unbiased),
        "measurable_edge_length_mm": _round(sum(e["length"] for e in measurable), 2),
        "unbiased_edge_length_mm": _round(sum(e["length"] for e in unbiased), 2),
        "feature_spread_rms_mm": _round(spread, 2),
        "min_normal_separation_deg": _round(min_separation, 2),
        "mirror_pairs_y": mirror_pairs(model),
        "unbiased_edge_length_mm_by_alignment": alignment_comparison(model),
        **chirality(model),
    }


def warnings_for(model: dict[str, Any], stats: dict[str, Any]) -> list[str]:
    out: list[str] = []
    facets = [f for f in model["facets"] if f["use_for_pose"]]
    carriers = stats["chirality_carrying_facets"]
    blind = stats["side_blind_pairs"]
    if not carriers:
        out.append(
            "the painted skin is mirror-symmetric about y = 0 and every mirror pair shares a colour, "
            "so colour alone never says which side of the rig a facet is on. The two anchor ids carry "
            "that, and a frame with no anchor visible must not be solved from facets alone."
        )
    elif blind:
        out.append(
            "colour resolves the side of y = 0 only in views that show one of {} ({:.0%} of painted "
            "area); {} are still same-coloured mirror twins and say nothing about which side they "
            "are on. So a facet-only pose is admissible only with a chirality carrier in view -- "
            "everything else still needs an anchor id.".format(
                ", ".join(carriers),
                stats["chirality_carrying_area_fraction"],
                ", ".join("/".join(pair) for pair in blind),
            )
        )
    black = [f["name"] for f in facets if f["colour"] == "black"]
    if black:
        out.append(
            "black facets ({}) cancel their own black border: the ink runs off into the fill, so every "
            "edge they touch is a one-sided step offset by a stroke width instead of a self-centring "
            "band. Repainting them a light colour would convert {} edges to unbiased.".format(
                ", ".join(black),
                sum(
                    1
                    for e in model["edges"]
                    if "black" in (e["colour_a"], e["colour_b"]) and e["profile"] != "band_centre"
                ),
            )
        )
    excluded = [f for f in model["facets"] if not f["use_for_pose"]]
    for facet in excluded:
        out.append(f"facet '{facet['name']}' excluded from pose: {facet['note']}")
    if stats["same_colour_shared_edges"]:
        out.append(
            "adjacent facets share a colour, so their common edge has no colour contrast: "
            f"{stats['same_colour_shared_edges']}"
        )
    by_alignment = stats["unbiased_edge_length_mm_by_alignment"]
    chosen = model["print"]["border"]["alignment"]
    best = max(by_alignment, key=lambda key: by_alignment[key])
    if by_alignment[chosen] < by_alignment[best]:
        out.append(
            f"the black stroke is specified '{chosen}', which leaves {by_alignment[chosen]:.0f} mm of "
            f"self-centring edge; printing it '{best}' would leave {by_alignment[best]:.0f} mm. If the "
            "stroke cannot be centred, border_width_mm stops being a comment and has to be measured, "
            "because the detector subtracts half of it from every one-sided landmark."
        )
    if any(a["paste_quadrant"] is None for a in model["anchors"]):
        out.append(
            "no anchor has its sticker rotation frozen; the detector enumerates four rotations per "
            "anchor per frame. Measure them once after assembly and write paste_quadrant."
        )
    pivot = model.get("pivot") or {}
    if pivot.get("socket_sphere_centre_mm") is None:
        out.append(
            "this STEP has no seating socket, so the rig frame is whatever the part studio's origin "
            "happens to be and the rig has no CAD-known TCP. Every downstream pivot number would then "
            "be an offset with no reference to compare it against."
        )
    elif float(pivot.get("offset_from_origin_mm") or 0.0) > 0.05:
        out.append(
            "the socket sphere centre sits {:.3f} mm off the part origin, so the rig frame is not the "
            "TCP frame after all: a pivot fit no longer has the known answer p_rig = 0, and the "
            "carrier->TCP constant is that offset rather than zero.".format(
                float(pivot["offset_from_origin_mm"])
            )
        )
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("step", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--anchor-ids",
        type=int,
        nargs="+",
        default=[30, 31],
        help=(
            "one ArUco id per unpainted pad, in +Y then +Z then +X order. Defaults avoid the ids "
            "already in use on this bench (0-5 on the BOX cube, 7/12/14/16/17 on the old rig, "
            "20-29 on the static scene markers)"
        ),
    )
    parser.add_argument("--dictionary", default="DICT_6X6_50")
    parser.add_argument("--pad-size-mm", type=float, default=60.0)
    parser.add_argument("--marker-size-mm", type=float, default=48.0)
    parser.add_argument("--pad-tolerance-mm", type=float, default=0.35)
    parser.add_argument(
        "--pad-face",
        action="append",
        default=[],
        type=int,
        metavar="ENTITY",
        help=(
            "treat this STEP face as an anchor pad even though the CAD paints it. Repeatable. "
            "The tabs on this carrier are double sided and the CAD colours one of the far sides, "
            "but a sticker goes on just as well as paint"
        ),
    )
    parser.add_argument(
        "--anchor-id",
        action="append",
        default=[],
        metavar="ENTITY=ID",
        help=(
            "pin one pad's marker id to a STEP face. Repeatable. Without it, ids from --anchor-ids "
            "go to the pads in sorted order, which renumbers the existing pads as soon as a new one "
            "is added -- and a printed sticker cannot be renumbered"
        ),
    )
    parser.add_argument(
        "--border-width-mm",
        type=float,
        default=1.0,
        help="printed black stroke width; not in the STEP, so state it here",
    )
    parser.add_argument(
        "--sticker-covers-pad",
        action="store_true",
        help=(
            "the ArUco sticker covers the whole 60 mm pad rather than just the marker; this stops a "
            "facet's black stroke from wrapping onto the pad and biases every pad-to-facet edge by "
            "half a stroke width"
        ),
    )
    parser.add_argument(
        "--border-alignment",
        choices=["inside", "centred", "outside"],
        default="centred",
        help=(
            "where the stroke sits relative to the CAD edge. 'centred' is the default because it "
            "makes every painted-against-bare-body edge self-centring, and so unbiased whatever the "
            "stroke width turns out to be; 'inside' pushes each of those landmarks half a stroke in"
        ),
    )
    parser.add_argument(
        "--repaint",
        action="append",
        default=[],
        metavar="FACET=COLOUR",
        help=(
            "paint a facet a colour the STEP does not carry, e.g. --repaint black_ny=green. "
            "Repeatable. FACET is the name the facet has under the STEP's own colour. Two uses: "
            "a black facet swallows its own border and makes every edge it touches one-sided, so "
            "repainting it light buys those edges back; and giving a mirror pair two different "
            "colours makes the painted skin chiral, which is the only thing besides an anchor id "
            "that can say which side of y = 0 a facet is on"
        ),
    )
    parser.add_argument(
        "--paste-quadrant",
        action="append",
        default=[],
        metavar="MARKER_ID=QUADRANT",
        help=(
            "freeze one sticker's rotation on its pad, in 90 deg steps (0-3). Repeatable. "
            "Without it the detector enumerates four rotations per anchor per frame, which is "
            "not only 4^n slower: on the 2026-09-09 bench capture about 2%% of anchor "
            "observations picked the wrong rotation, and a wrong rotation does not fail, it "
            "returns a confident pose tens of millimetres out. Measure it once on the built rig"
        ),
    )
    parser.add_argument("--carrier-id", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    step = read_step(args.step)
    model = build_model(
        step,
        anchor_ids=list(args.anchor_ids),
        dictionary=args.dictionary,
        pad_size_mm=args.pad_size_mm,
        marker_size_mm=args.marker_size_mm,
        pad_tolerance_mm=args.pad_tolerance_mm,
        pad_faces=list(args.pad_face),
        anchor_id_by_face={
            int(item.split("=", 1)[0]): int(item.split("=", 1)[1]) for item in args.anchor_id
        },
        border_width_mm=args.border_width_mm,
        border_alignment=args.border_alignment,
        sticker_covers_pad=bool(args.sticker_covers_pad),
        carrier_id=args.carrier_id or args.step.stem,
        repaint=dict(item.split("=", 1) for item in args.repaint),
        paste_quadrant_by_id={
            int(item.split("=", 1)[0]): int(item.split("=", 1)[1]) for item in args.paste_quadrant
        },
    )
    stats = diagnostics(model)
    model["diagnostics"] = stats
    model["warnings"] = warnings_for(model, stats)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(model, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[INFO] wrote {args.out}")
    print(
        "[INFO] {num_anchors} anchors, {num_facets_for_pose} facets for pose "
        "({num_facets_excluded} excluded), {num_measurable_edges}/{num_edges} edges measurable, "
        "{num_unbiased_edges} unbiased".format(**stats)
    )
    print(
        f"[INFO] measurable edge length {stats['measurable_edge_length_mm']} mm "
        f"(unbiased {stats['unbiased_edge_length_mm']} mm), feature spread "
        f"{stats['feature_spread_rms_mm']} mm rms, closest facet normals "
        f"{stats['min_normal_separation_deg']} deg apart"
    )
    for anchor in model["anchors"]:
        print(
            f"[INFO] anchor {anchor['name']}: id {anchor['marker_id']} on face #{anchor['face_entity_id']}"
        )
    for facet in model["facets"]:
        flag = "" if facet["use_for_pose"] else "  [excluded]"
        print(
            f"[INFO] facet {facet['name']:<12} {facet['colour']:<6} "
            f"{facet['num_vertices']}-gon {facet['area']:>8.1f} mm2 face #{facet['face_entity_id']}{flag}"
        )
    for warning in model["warnings"]:
        print(f"[WARN] {warning}")


if __name__ == "__main__":
    main()
