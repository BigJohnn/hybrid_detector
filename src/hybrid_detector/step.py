"""Read an AP242 STEP B-rep well enough to describe a painted marker rig.

The rig descriptor needs four things out of a STEP file: which faces exist,
what polygon bounds each planar face, which faces share an edge, and what
colour the CAD author painted each face.  Every heavy CAD kernel can do this,
but binding one (``cadquery``/OCC) buys a large dependency and, worse, forces
the caller to assume the kernel enumerates faces in the same order the STEP
file lists ``ADVANCED_FACE`` entities -- an assumption that a matching *count*
does not verify, and that silently mislabels colours when it breaks.  Reading
the file directly keeps the STEP entity id attached to the geometry it came
from, so a face and its ``OVER_RIDING_STYLED_ITEM`` colour can never drift
apart.

Only what a painted-facet rig needs is parsed:

* planar faces, as ordered polygons wound counter-clockwise about the face's
  outward normal (that is the surface normal flipped by ``same_sense``);
* the ``EDGE_CURVE`` id behind every oriented edge, which makes face adjacency
  an exact identity test instead of a coordinate-rounding heuristic;
* per-face appearance, resolved through the
  ``STYLED_ITEM``/``PRESENTATION_STYLE_ASSIGNMENT`` chain;
* the spheres and cylinders, because the rig's frame is defined by a socket
  sphere centre and a facet can be pierced by a bore.

Curved faces are recorded but not tessellated; their loops keep whatever
vertices they have so a caller can still see they exist.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "StepEntity",
    "StepFace",
    "StepModel",
    "read_step",
    "semantic_colour",
]

_UNIT_SCALE_TO_MM = {
    "MILLI.,.METRE": 1.0,
    "$,.METRE": 1000.0,
    "CENTI.,.METRE": 10.0,
}


@dataclass(frozen=True)
class StepEntity:
    entity_id: int
    keyword: str
    body: str


@dataclass(frozen=True)
class StepFace:
    """One ``ADVANCED_FACE``, with the loops resolved into coordinates."""

    entity_id: int
    surface_type: str
    same_sense: bool
    colour: Any | None
    colour_is_override: bool
    normal: np.ndarray | None
    outer_polygon: np.ndarray | None
    inner_polygons: tuple[np.ndarray, ...]
    outer_edge_ids: tuple[int, ...]
    inner_edge_ids: tuple[tuple[int, ...], ...]
    outer_is_polyline: bool
    area: float | None
    centroid: np.ndarray | None
    surface_params: dict[str, Any] = field(default_factory=dict)

    @property
    def is_plane(self) -> bool:
        return self.surface_type == "plane"

    @property
    def edge_ids(self) -> frozenset[int]:
        ids: set[int] = set(self.outer_edge_ids)
        for loop in self.inner_edge_ids:
            ids.update(loop)
        return frozenset(ids)


@dataclass(frozen=True)
class StepModel:
    path: Path
    sha256: str
    header: dict[str, Any]
    units: str
    length_scale_to_mm: float
    body_colour: Any | None
    faces: tuple[StepFace, ...]

    def face(self, entity_id: int) -> StepFace:
        for f in self.faces:
            if f.entity_id == int(entity_id):
                return f
        raise KeyError(entity_id)

    @property
    def planar_faces(self) -> tuple[StepFace, ...]:
        return tuple(f for f in self.faces if f.is_plane)

    def shared_edge_ids(self, a: StepFace, b: StepFace) -> frozenset[int]:
        return a.edge_ids & b.edge_ids


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _split_statements(text: str) -> list[str]:
    """Split a STEP section on ``;`` while respecting quoted strings."""
    out: list[str] = []
    start = 0
    in_string = False
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if in_string:
            if c == "'":
                if i + 1 < n and text[i + 1] == "'":
                    i += 1
                else:
                    in_string = False
        elif c == "'":
            in_string = True
        elif c == ";":
            out.append(text[start:i])
            start = i + 1
        i += 1
    tail = text[start:].strip()
    if tail:
        out.append(tail)
    return out


def split_arguments(body: str) -> list[str]:
    """Split one entity's argument list on top-level commas."""
    out: list[str] = []
    depth = 0
    in_string = False
    cur: list[str] = []
    i = 0
    n = len(body)
    while i < n:
        c = body[i]
        if in_string:
            cur.append(c)
            if c == "'":
                if i + 1 < n and body[i + 1] == "'":
                    cur.append("'")
                    i += 1
                else:
                    in_string = False
        elif c == "'":
            in_string = True
            cur.append(c)
        elif c == "(":
            depth += 1
            cur.append(c)
        elif c == ")":
            depth -= 1
            cur.append(c)
        elif c == "," and depth == 0:
            out.append("".join(cur).strip())
            cur = []
        else:
            cur.append(c)
        i += 1
    out.append("".join(cur).strip())
    return out


def _ref(token: str) -> int | None:
    token = token.strip()
    return int(token[1:]) if token.startswith("#") else None


def _refs(token: str) -> list[int]:
    return [int(x) for x in re.findall(r"#(\d+)", token)]


def _unit_vector(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= 0.0:
        raise ValueError("zero-length direction in STEP file")
    return v / n


class _Reader:
    def __init__(self, text: str) -> None:
        header_text = text.split("HEADER;", 1)[1].split("ENDSEC;", 1)[0]
        data_text = text.split("DATA;", 1)[1].rsplit("ENDSEC;", 1)[0]
        self.header = self._parse_header(header_text)
        self.entities: dict[int, StepEntity] = {}
        for statement in _split_statements(data_text):
            statement = statement.strip()
            if not statement.startswith("#"):
                continue
            head, _, rest = statement.partition("=")
            entity_id = int(head.strip()[1:])
            body = re.sub(r"\s+", "", rest.strip())
            match = re.match(r"^([A-Z_0-9]+)\((.*)\)$", body, flags=re.S)
            if match is None:
                # Complex (AND-combined) instance, e.g. the unit context.
                self.entities[entity_id] = StepEntity(entity_id, "_COMPLEX", body)
                continue
            self.entities[entity_id] = StepEntity(entity_id, match.group(1), match.group(2))

    @staticmethod
    def _parse_header(header_text: str) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for statement in _split_statements(header_text):
            match = re.match(r"^\s*([A-Z_0-9]+)\s*\((.*)\)\s*$", statement, flags=re.S)
            if match is None:
                continue
            args = split_arguments(re.sub(r"/\*.*?\*/", "", match.group(2), flags=re.S))
            out[match.group(1)] = [a.strip().strip("'") for a in args]
        return out

    def keyword(self, entity_id: int) -> str:
        return self.entities[entity_id].keyword

    def args(self, entity_id: int) -> list[str]:
        return split_arguments(self.entities[entity_id].body)

    # -- geometry -------------------------------------------------------
    def point(self, entity_id: int) -> np.ndarray:
        coords = split_arguments(self.args(entity_id)[1][1:-1])
        return np.array([float(x) for x in coords], dtype=np.float64)

    def direction(self, entity_id: int) -> np.ndarray:
        coords = split_arguments(self.args(entity_id)[1][1:-1])
        return _unit_vector(np.array([float(x) for x in coords], dtype=np.float64))

    def placement(self, entity_id: int) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        a = self.args(entity_id)
        origin = self.point(_ref(a[0 + 1]))
        axis = self.direction(_ref(a[2])) if a[2] != "$" else None
        ref_dir = self.direction(_ref(a[3])) if len(a) > 3 and a[3] != "$" else None
        return origin, axis, ref_dir

    def vertex_point(self, entity_id: int) -> np.ndarray:
        return self.point(_ref(self.args(entity_id)[1]))

    def loop(self, loop_id: int) -> tuple[np.ndarray, tuple[int, ...], bool]:
        """Ordered start-vertices of an EDGE_LOOP, its EDGE_CURVE ids, and polyline-ness."""
        oriented = _refs(self.args(loop_id)[1])
        points: list[np.ndarray] = []
        edge_ids: list[int] = []
        polyline = True
        for oe in oriented:
            oa = self.args(oe)
            edge_curve = _ref(oa[3])
            forward = oa[4] == ".T."
            ea = self.args(edge_curve)
            start = self.vertex_point(_ref(ea[0 + 1]))
            end = self.vertex_point(_ref(ea[2]))
            if not forward:
                start, end = end, start
            curve = _ref(ea[3])
            keyword = self.keyword(curve)
            if keyword != "LINE":
                polyline = False
            if keyword == "CIRCLE" and len(oriented) == 1:
                # A bore is one edge that starts and ends at the same vertex, so
                # walking start points yields a single point and the loop reads
                # as empty area.  That silently turns a hole into nothing: the
                # occluder model stops filling it, and a face with a hole
                # through the middle of it looks like solid material to anything
                # asking whether a sticker fits.  Approximate it instead.
                points.extend(self.circle_points(curve))
                edge_ids.append(int(edge_curve))
                continue
            points.append(start)
            edge_ids.append(int(edge_curve))
        return np.asarray(points, dtype=np.float64), tuple(edge_ids), polyline

    def circle_points(self, circle_id: int, segments: int = 24) -> list[np.ndarray]:
        """A CIRCLE as a closed polygon in its own plane."""
        arguments = self.args(circle_id)
        placement = self.args(_ref(arguments[1]))
        radius = float(arguments[2])
        centre = self.point(_ref(placement[1]))
        axis = _unit_vector(self.direction(_ref(placement[2])))
        reference = _unit_vector(self.direction(_ref(placement[3])))
        u = _unit_vector(reference - float(np.dot(reference, axis)) * axis)
        v = np.cross(axis, u)
        angles = np.linspace(0.0, 2.0 * np.pi, int(segments), endpoint=False)
        return [centre + radius * (np.cos(a) * u + np.sin(a) * v) for a in angles]

    # -- appearance -----------------------------------------------------
    def _colour_of(self, colour_id: int) -> Any:
        entity = self.entities[colour_id]
        if entity.keyword == "DRAUGHTING_PRE_DEFINED_COLOUR":
            return split_arguments(entity.body)[0].strip("'").lower()
        if entity.keyword == "COLOUR_RGB":
            return tuple(round(float(x), 6) for x in split_arguments(entity.body)[1:])
        raise ValueError(f"#{colour_id} is not a colour ({entity.keyword})")

    def _style_colour(self, style_assignment_id: int) -> Any | None:
        """PRESENTATION_STYLE_ASSIGNMENT -> ... -> FILL_AREA_STYLE_COLOUR -> colour."""
        seen: set[int] = set()
        stack = [int(style_assignment_id)]
        while stack:
            current = stack.pop()
            if current in seen or current not in self.entities:
                continue
            seen.add(current)
            entity = self.entities[current]
            if entity.keyword in {"DRAUGHTING_PRE_DEFINED_COLOUR", "COLOUR_RGB"}:
                return self._colour_of(current)
            stack.extend(_refs(entity.body))
        return None

    def appearances(self) -> tuple[Any | None, dict[int, Any]]:
        body_colour: Any | None = None
        overrides: dict[int, Any] = {}
        for entity in self.entities.values():
            if entity.keyword == "STYLED_ITEM":
                a = split_arguments(entity.body)
                styles = _refs(a[1])
                if styles:
                    body_colour = self._style_colour(styles[0])
            elif entity.keyword == "OVER_RIDING_STYLED_ITEM":
                a = split_arguments(entity.body)
                styles = _refs(a[1])
                target = _ref(a[2])
                if styles and target is not None:
                    overrides[int(target)] = self._style_colour(styles[0])
        return body_colour, overrides

    def units(self) -> tuple[str, float]:
        for entity in self.entities.values():
            if entity.keyword != "_COMPLEX":
                continue
            match = re.search(r"SI_UNIT\(([^)]*)\)", entity.body)
            if match is None or "METRE" not in entity.body:
                continue
            key = match.group(1).strip()
            if "LENGTH_UNIT" not in entity.body:
                continue
            scale = _UNIT_SCALE_TO_MM.get(key)
            if scale is not None:
                return key.replace(".", "").replace("$,", "").lower() or "mm", scale
        return "mm", 1.0


def _polygon_area_normal(points: np.ndarray) -> tuple[float, np.ndarray]:
    accumulator = np.zeros(3, dtype=np.float64)
    count = len(points)
    for i in range(count):
        accumulator += np.cross(points[i], points[(i + 1) % count])
    norm = float(np.linalg.norm(accumulator))
    if norm <= 0.0:
        return 0.0, np.zeros(3, dtype=np.float64)
    return norm / 2.0, accumulator / norm


def _wind_ccw(
    points: np.ndarray, edge_ids: tuple[int, ...], normal: np.ndarray
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Wind a loop counter-clockwise about ``normal``, edge ids following along.

    Callers rely on segment ``k`` running ``points[k] -> points[k + 1]`` and
    being the curve ``edge_ids[k]``; reversing the points without reversing the
    ids the same way would quietly re-label every edge of a clockwise face.
    """
    if len(points) < 3:
        return points, edge_ids
    _, poly_normal = _polygon_area_normal(points)
    if float(np.dot(poly_normal, normal)) >= 0.0:
        return points, edge_ids
    count = len(points)
    if len(edge_ids) != count:
        # A loop whose points were interpolated rather than walked -- a bore
        # approximated as a polygon -- has no per-segment id to keep in step,
        # so there is nothing to re-index.
        return points[::-1].copy(), tuple(reversed(edge_ids))
    reversed_ids = (
        tuple(edge_ids[(count - 2 - j) % count] for j in range(count)) if edge_ids else edge_ids
    )
    return points[::-1].copy(), reversed_ids


def _planar_polygon_area(points: np.ndarray, normal: np.ndarray) -> float:
    accumulator = np.zeros(3, dtype=np.float64)
    count = len(points)
    for i in range(count):
        accumulator += np.cross(points[i], points[(i + 1) % count])
    return abs(float(np.dot(accumulator, normal))) / 2.0


def read_step(path: str | Path) -> StepModel:
    """Parse the planar B-rep and per-face appearance of a STEP file."""
    path = Path(path)
    reader = _Reader(path.read_text(encoding="utf-8", errors="replace"))
    body_colour, overrides = reader.appearances()
    units, scale = reader.units()

    faces: list[StepFace] = []
    for entity in reader.entities.values():
        if entity.keyword != "ADVANCED_FACE":
            continue
        a = split_arguments(entity.body)
        bound_ids = _refs(a[1])
        surface_id = _ref(a[2])
        same_sense = a[3] == ".T."
        surface_keyword = reader.keyword(surface_id)
        surface_type = {
            "PLANE": "plane",
            "CYLINDRICAL_SURFACE": "cylinder",
            "SPHERICAL_SURFACE": "sphere",
            "CONICAL_SURFACE": "cone",
            "TOROIDAL_SURFACE": "torus",
            "B_SPLINE_SURFACE_WITH_KNOTS": "bspline",
        }.get(surface_keyword, surface_keyword.lower())

        normal: np.ndarray | None = None
        surface_params: dict[str, Any] = {}
        surface_args = reader.args(surface_id)
        if surface_type == "plane":
            origin, axis, _ = reader.placement(_ref(surface_args[1]))
            normal = axis if same_sense else -axis
            surface_params = {"origin": origin, "axis": axis}
        elif surface_type in {"cylinder", "sphere", "cone"}:
            origin, axis, _ = reader.placement(_ref(surface_args[1]))
            surface_params = {"origin": origin, "axis": axis, "radius": float(surface_args[2])}

        loops = []
        for bound_id in bound_ids:
            ba = reader.args(bound_id)
            points, edge_ids, polyline = reader.loop(_ref(ba[0 + 1]))
            area = (
                _planar_polygon_area(points, normal)
                if (normal is not None and len(points) >= 3)
                else 0.0
            )
            loops.append((area, points, edge_ids, polyline))
        # Onshape writes FACE_BOUND for outer bounds too, so the outer loop is
        # identified by area rather than by the FACE_OUTER_BOUND keyword.
        loops.sort(key=lambda item: item[0], reverse=True)

        outer_area, outer_points, outer_edges, outer_polyline = loops[0]
        outer_polygon = None
        centroid = None
        area: float | None = None
        if normal is not None and len(outer_points) >= 3 and outer_polyline:
            outer_polygon, outer_edges = _wind_ccw(outer_points, outer_edges, normal)
            centroid = outer_polygon.mean(axis=0)
            area = outer_area - sum(item[0] for item in loops[1:])
        elif normal is not None and len(outer_points) >= 3:
            centroid = outer_points.mean(axis=0)

        faces.append(
            StepFace(
                entity_id=entity.entity_id,
                surface_type=surface_type,
                same_sense=same_sense,
                colour=overrides.get(entity.entity_id, None),
                colour_is_override=entity.entity_id in overrides,
                normal=normal,
                outer_polygon=outer_polygon,
                inner_polygons=tuple(
                    _wind_ccw(item[1], item[2], normal)[0]
                    if (normal is not None and len(item[1]) >= 3)
                    else item[1]
                    for item in loops[1:]
                ),
                outer_edge_ids=outer_edges,
                inner_edge_ids=tuple(item[2] for item in loops[1:]),
                outer_is_polyline=outer_polyline,
                area=area,
                centroid=centroid,
                surface_params=surface_params,
            )
        )

    faces.sort(key=lambda f: f.entity_id)
    return StepModel(
        path=path,
        sha256=sha256_of(path),
        header=reader.header,
        units=units,
        length_scale_to_mm=scale,
        body_colour=body_colour,
        faces=tuple(faces),
    )


_GREY_TOLERANCE = 0.06


def semantic_colour(colour: Any | None) -> str | None:
    """Map a STEP appearance to the paint name the detector reasons about."""
    if colour is None:
        return None
    if isinstance(colour, str):
        return colour.strip().lower()
    rgb = np.asarray(colour, dtype=np.float64).reshape(3)
    r, g, b = rgb
    if float(rgb.max() - rgb.min()) < _GREY_TOLERANCE:
        return "white" if rgb.mean() > 0.85 else ("black" if rgb.mean() < 0.15 else "grey")
    if r >= g and r >= b:
        return "red" if g < 0.5 else "yellow"
    if g >= r and g >= b:
        return "green"
    return "blue"
