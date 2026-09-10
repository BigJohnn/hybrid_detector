#!/usr/bin/env python3
"""Draw the decal installation sheet from the descriptor the detector reads.

Usage::

    python -m hybrid_detector.cli.render_install_guide \
        assets/models/hybrid_carrier_v1_20260907.json \
        --manifest assets/print/rig0907_patterns_RGB_exact1mm_manifest.json \
        --step assets/cad/rig0907_rgb.step \
        --out assets/print/rig0907_RGB_installation_guide.png

A guide drawn by projecting every label and hoping the right ones land in
front puts a back facet's name on the face in front of it, and the assembler
cannot tell which of the two labels is the lie.  So labels here are placed
through the same depth buffer the renderer uses: a facet is labelled only where
it is *the visible surface*, at the deepest interior point of that visible
region.  A facet with no such region in a view is simply not labelled there.

Views are chosen, not guessed.  Candidate directions are swept over the sphere,
each facet is scored by how many pixels of it a view actually shows, and views
are taken greedily until every facet and every anchor has one view that shows
it well.  If the sweep cannot cover something the guide says so rather than
quietly omitting it -- an unlabelled facet is a decal in a bag with no home.

Decals the manifest prints but the descriptor does not carry (a facet the
builder dropped) are listed as *not applied*, keeping the print ids on the
sheet aligned with the printed page numbers instead of renumbering them.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from hybrid_detector.calibration import CameraCalibration
from hybrid_detector.detector import (
    HybridCarrierModel,
    _inverse_depth_plane,
    project_rig,
    transform,
)
from hybrid_detector.render import PaintPalette, render_carrier
from hybrid_detector.step import _split_statements

CJK_FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
MONO_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
INK = {"R": (230, 0, 18), "G": (0, 166, 81), "B": (0, 87, 184)}


def step_face_order(path: Path) -> list[int]:
    """ADVANCED_FACE entity ids in file order -- the manifest's index space."""
    data = path.read_text().split("DATA;", 1)[1].rsplit("ENDSEC;", 1)[0]
    out: list[int] = []
    for raw in _split_statements(data):
        raw = raw.strip()
        if not raw.startswith("#"):
            continue
        prefix, _, body = raw.partition("=")
        if re.sub(r"\s+", "", body).startswith("ADVANCED_FACE("):
            out.append(int(prefix.strip()[1:]))
    return out


def face_index_buffer(model: HybridCarrierModel, camera: CameraCalibration, T_cam_rig: np.ndarray):
    """Which STEP face the camera sees at each pixel -- the renderer's depth pass."""
    height, width = int(camera.height), int(camera.width)
    inverse_depth = np.full((height, width), -np.inf, dtype=np.float64)
    face_index = np.full((height, width), -1, dtype=np.int64)
    grid_v, grid_u = np.mgrid[0:height, 0:width].astype(np.float64)
    for index, tri in enumerate(model.occluder_triangles):
        cam = transform(T_cam_rig, tri)
        if np.any(cam[:, 2] <= 1e-6):
            continue
        if float(np.dot(np.cross(cam[1] - cam[0], cam[2] - cam[0]), cam.mean(axis=0))) >= 0.0:
            continue
        uv = project_rig(camera, T_cam_rig, tri)
        if not np.isfinite(uv).all():
            continue
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [np.round(uv).astype(np.int32)], 255)
        if not np.any(mask):
            continue
        plane = _inverse_depth_plane(uv, cam[:, 2])
        if plane is None:
            continue
        candidate = plane[0] * grid_u + plane[1] * grid_v + plane[2]
        nearer = (mask > 0) & (candidate > inverse_depth)
        inverse_depth[nearer] = candidate[nearer]
        face_index[nearer] = int(model.occluder_face_ids[index])
    return face_index


def look_at(
    centre: np.ndarray, azimuth_deg: float, elevation_deg: float, distance: float
) -> np.ndarray:
    az, el = np.radians(azimuth_deg), np.radians(elevation_deg)
    eye = centre + distance * np.array(
        [np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)]
    )
    forward = centre - eye
    forward /= np.linalg.norm(forward)
    world_up = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(forward, world_up))) > 0.999:
        world_up = np.array([1.0, 0.0, 0.0])
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    T_cam_rig = np.eye(4)
    T_cam_rig[:3, :3] = np.column_stack([right, down, forward]).T
    T_cam_rig[:3, 3] = -T_cam_rig[:3, :3] @ eye
    return T_cam_rig


def label_anchor_point(mask: np.ndarray, half: tuple[int, int]) -> tuple[tuple[int, int], bool]:
    """Deepest interior point of a mask, and whether a label of that size fits."""
    distance = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 3)
    _, radius, _, point = cv2.minMaxLoc(distance)
    needed = float(np.hypot(*half))
    return (int(point[0]), int(point[1])), bool(radius >= needed)


def draw_nose_gizmo(draw, camera, T_cam_rig, centre, span, size, font) -> None:
    """Which way the nose points in this view, so the part can be held to match."""
    axis = project_rig(
        camera, T_cam_rig, np.array([centre, centre + np.array([0.35 * span, 0.0, 0.0])])
    )
    if not np.isfinite(axis).all():
        return
    origin = np.array([size - 96.0, 84.0])
    draw.rounded_rectangle(
        [size - 168, 22, size - 24, 148],
        radius=10,
        fill=(255, 255, 255),
        outline=(215, 215, 215),
        width=2,
    )
    direction = axis[1] - axis[0]
    length = float(np.linalg.norm(direction))
    if length > 3.0:
        unit = direction / length
        tip = origin + unit * 40.0
        tail = origin - unit * 40.0
        draw.line([tuple(tail), tuple(tip)], fill=(110, 110, 110), width=4)
        perpendicular = np.array([-unit[1], unit[0]]) * 9.0
        head = tip - unit * 15.0
        draw.polygon(
            [tuple(tip), tuple(head + perpendicular), tuple(head - perpendicular)],
            fill=(110, 110, 110),
        )
    else:  # end-on: the nose points at or away from the reader
        draw.ellipse(
            [origin[0] - 14, origin[1] - 14, origin[0] + 14, origin[1] + 14],
            outline=(110, 110, 110),
            width=4,
        )
        draw.ellipse(
            [origin[0] - 4, origin[1] - 4, origin[0] + 4, origin[1] + 4], fill=(110, 110, 110)
        )
    draw.text(
        (origin[0], origin[1] + 44), "+X 鼻子 nose", font=font, fill=(120, 120, 120), anchor="mm"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("descriptor", type=Path, help="the carrier descriptor to draw")
    parser.add_argument(
        "--manifest", type=Path, required=True, help="print manifest, for the decal ids"
    )
    parser.add_argument(
        "--step", type=Path, required=True, help="the STEP the manifest indexes into"
    )
    parser.add_argument("--out", type=Path, required=True, help="PNG to write")
    parser.add_argument("--tile", type=int, default=680, help="pixels per view")
    parser.add_argument("--max-views", type=int, default=6)
    parser.add_argument(
        "--min-face-on",
        type=float,
        default=0.55,
        help="cosine between a face normal and the eye below which a view does not count as showing it",
    )
    parser.add_argument(
        "--min-visible-px",
        type=int,
        default=1800,
        help="a view only counts as showing a facet if this many of its pixels are visible",
    )
    args = parser.parse_args(argv)

    model = HybridCarrierModel.from_json(args.descriptor)
    manifest = json.loads(args.manifest.read_text())
    faces = step_face_order(args.step)

    # decal id <- position on the printed sheet, kept even where a decal is dead
    printed = sorted(int(k) for k in manifest["color_faces"])
    decal_of_face = {faces[idx]: f"P{n + 1:02d}" for n, idx in enumerate(printed)}
    # Where the sheet's outlines are mirrored, the decal that actually fits a face is
    # its twin's, and the assembler is holding a sticker whose own caption names the
    # other face.  This guide has to name the sticker in the hand, so the manifest's
    # as-applied map wins over the positional numbering.
    mirror = manifest.get("sheet_outline_mirror") or {}
    decal_of_face.update(
        {
            faces[int(idx)]: name
            for idx, name in (mirror.get("decal_applied_to_face_index") or {}).items()
        }
    )
    colour_of_face = {faces[int(k)]: v for k, v in manifest["color_faces"].items()}
    anchor_of_face = {faces[int(k)]: int(v) for k, v in manifest["anchors"].items()}

    facet_by_face = {int(f.face_entity_id): f for f in model.facets}
    anchor_by_face = {int(a.face_entity_id): a for a in model.anchors}
    live = set(facet_by_face) | set(anchor_by_face)
    dead = [(decal_of_face[f], f) for f in sorted(decal_of_face) if f not in live]

    label_of: dict[int, str] = {}
    for face in facet_by_face:
        label_of[face] = decal_of_face.get(face, facet_by_face[face].name)
    for face, marker_id in anchor_of_face.items():
        if face in anchor_by_face:
            label_of[face] = f"A{marker_id}"

    size = int(args.tile)
    focal = size * 1.45
    camera = CameraCalibration(
        "guide",
        "guide",
        size,
        size,
        np.array([[focal, 0, size / 2], [0, focal, size / 2], [0, 0, 1]], dtype=np.float64),
        np.zeros(5),
        np.eye(4),
        "rational",
        0.2,
    )
    points = np.vstack(
        [f.polygon_m for f in model.facets] + [a.pad_corners_m for a in model.anchors]
    )
    centre = 0.5 * (points.min(axis=0) + points.max(axis=0))
    span = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
    distance = 1.55 * span

    # -- choose views by what they actually show ---------------------------
    normal_of = {int(f.face_entity_id): np.asarray(f.normal, float) for f in model.facets}
    normal_of.update({int(a.face_entity_id): np.asarray(a.normal, float) for a in model.anchors})
    candidates = [
        (az, el) for el in (-55.0, -25.0, 5.0, 35.0, 65.0) for az in np.arange(-180.0, 180.0, 30.0)
    ] + [(0.0, 88.0), (0.0, -88.0)]
    seen: dict[tuple[float, float], dict[int, int]] = {}
    for az, el in candidates:
        T_cam_rig = look_at(centre, az, el, distance)
        buffer = face_index_buffer(model, camera, T_cam_rig)
        eye = -T_cam_rig[:3, :3].T @ T_cam_rig[:3, 3]
        ids, counts = np.unique(buffer[buffer >= 0], return_counts=True)
        keep_here: dict[int, int] = {}
        for face, count in zip(ids, counts, strict=True):
            face = int(face)
            if face not in label_of or count < args.min_visible_px:
                continue
            # A sliver is visible without being *shown*.  Requiring the face to
            # be square-on to the eye is what stops the chooser from covering
            # the whole rig from three high views and calling every edge-on
            # facet labelled.
            towards = eye - np.asarray(
                facet_by_face[face].centroid_m
                if face in facet_by_face
                else anchor_by_face[face].centre_m,
                dtype=float,
            )
            towards /= np.linalg.norm(towards) or 1.0
            if float(np.dot(normal_of[face], towards)) < args.min_face_on:
                continue
            keep_here[face] = int(count)
        seen[(az, el)] = keep_here

    chosen: list[tuple[float, float]] = []
    uncovered = set(label_of)
    while uncovered and len(chosen) < args.max_views:
        best = max(candidates, key=lambda v: (len(uncovered & set(seen[v])), sum(seen[v].values())))
        if not (uncovered & set(seen[best])):
            break
        chosen.append(best)
        uncovered -= set(seen[best])
    if uncovered:
        print(
            f"[WARN] no view in the sweep shows {len(uncovered)}: {sorted(label_of[f] for f in uncovered)}"
        )
    # each face is labelled once, in the chosen view that shows most of it
    assigned: dict[tuple[float, float], list[int]] = {v: [] for v in chosen}
    for face in label_of:
        options = [(seen[v].get(face, 0), v) for v in chosen]
        count, view = max(options)
        if count > 0:
            assigned[view].append(face)

    # -- draw --------------------------------------------------------------
    palette = PaintPalette(background=(250, 250, 250))
    tiles: list[Image.Image] = []
    tag = ImageFont.truetype(MONO_FONT, 26)
    caption = ImageFont.truetype(CJK_FONT, 20)
    for number, (az, el) in enumerate(chosen, start=1):
        T_cam_rig = look_at(centre, az, el, distance)
        frame = render_carrier(model, camera, T_cam_rig, palette=palette, marker_pixels=320)
        buffer = face_index_buffer(model, camera, T_cam_rig)
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(image)
        painted = np.argwhere(buffer >= 0)
        silhouette_centre = (
            painted.mean(axis=0)[::-1] if len(painted) else np.array([size / 2, size / 2])
        )
        draw_nose_gizmo(draw, camera, T_cam_rig, centre, span, size, caption)
        for face in sorted(assigned[(az, el)], key=lambda f: label_of[f]):
            text = label_of[face]
            box = draw.textbbox((0, 0), text, font=tag)
            half = ((box[2] - box[0]) // 2 + 9, (box[3] - box[1]) // 2 + 7)
            mask = (buffer == face).astype(np.uint8) * 255
            spot, fits = label_anchor_point(mask, half)
            fits = fits and face not in anchor_by_face  # never cover a marker pattern
            target = spot
            if not fits:
                # Step the label off the part and lead a line back to it, far
                # enough that it clears the face it names -- a leader that ends
                # inside its own facet is just an offset label.
                away = np.array(spot, dtype=np.float64) - silhouette_centre
                norm = float(np.linalg.norm(away))
                if norm < 1.0:
                    away, norm = np.array([0.0, 1.0]), 1.0
                away /= norm
                for push in range(60, 260, 10):
                    trial = np.array(spot, dtype=np.float64) + away * push
                    box_x = (int(trial[0] - half[0]), int(trial[0] + half[0]))
                    box_y = (int(trial[1] - half[1]), int(trial[1] + half[1]))
                    if not (
                        box_x[0] >= 26
                        and box_x[1] < size - 26
                        and box_y[0] >= 26
                        and box_y[1] < size - 26
                    ):
                        break
                    if not mask[box_y[0] : box_y[1], box_x[0] : box_x[1]].any():
                        target = (int(trial[0]), int(trial[1]))
                        break
                    target = (int(trial[0]), int(trial[1]))
                draw.line([spot, target], fill=(20, 20, 20), width=2)
                draw.ellipse(
                    [spot[0] - 4, spot[1] - 4, spot[0] + 4, spot[1] + 4], fill=(20, 20, 20)
                )
            draw.rounded_rectangle(
                [
                    target[0] - half[0],
                    target[1] - half[1],
                    target[0] + half[0],
                    target[1] + half[1],
                ],
                radius=7,
                fill=(255, 255, 255),
                outline=(20, 20, 20),
                width=2,
            )
            draw.text(target, text, font=tag, fill=(15, 15, 15), anchor="mm")
        # where the nose points, so the part can be held the same way as the drawing
        draw.rectangle([0, 0, size - 1, size - 1], outline=(200, 200, 200), width=2)
        draw.rectangle([0, 0, 250, 32], fill=(30, 30, 30))
        draw.text(
            (10, 16),
            f"VIEW {number}   az {az:+.0f}   el {el:+.0f}",
            font=caption,
            fill=(255, 255, 255),
            anchor="lm",
        )
        tiles.append(image)

    columns = 2 if len(tiles) <= 4 else 3
    rows = (len(tiles) + columns - 1) // columns
    legend_height = 150 + 30 * (len(printed) + len(model.anchors) + 7)
    sheet = Image.new("RGB", (columns * size, rows * size + legend_height), (255, 255, 255))
    for index, tile in enumerate(tiles):
        sheet.paste(tile, ((index % columns) * size, (index // columns) * size))

    draw = ImageDraw.Draw(sheet)
    title = ImageFont.truetype(CJK_FONT, 30)
    body = ImageFont.truetype(CJK_FONT, 20)
    mono = ImageFont.truetype(MONO_FONT, 19)
    y = rows * size + 22
    draw.text(
        (26, y),
        f"{model.carrier_id}  贴纸安装图 / decal installation",
        font=title,
        fill=(15, 15, 15),
    )
    y += 42
    for line in (
        "由描述符生成，不是手工标注：每个标签只画在该面自己可见的像素上（深度缓冲），背面的标签不会投到正面。",
        "Generated from the descriptor; every label is depth-tested against the face it names.",
    ):
        draw.text((26, y), line, font=body, fill=(70, 70, 70))
        y += 27
    y += 8
    view_of_face = {face: n for n, view in enumerate(chosen, start=1) for face in assigned[view]}
    for text, x in (
        ("decal", 26),
        ("colour", 100),
        ("view", 400),
        ("CAD face", 470),
        ("facet name", 590),
    ):
        draw.text((x, y), text, font=mono, fill=(120, 120, 120))
    y += 28
    # sorted by decal id, because the assembler is holding a numbered sticker and
    # looking for where it goes -- not holding a face and looking for its number
    for index in sorted(printed, key=lambda i: decal_of_face[faces[i]]):
        face = faces[index]
        key = colour_of_face[face]
        swatch = INK[key]
        alive = face in facet_by_face
        draw.text(
            (26, y), decal_of_face[face], font=mono, fill=(15, 15, 15) if alive else (170, 170, 170)
        )
        draw.rectangle(
            [100, y + 2, 124, y + 18],
            fill=swatch if alive else (235, 235, 235),
            outline=(150, 150, 150),
        )
        draw.text(
            (132, y),
            f"{key} {manifest['colors'][key]}",
            font=mono,
            fill=(90, 90, 90) if alive else (180, 180, 180),
        )
        if alive:
            draw.text((400, y), f"{view_of_face.get(face, '-')}", font=mono, fill=(15, 15, 15))
            draw.text((470, y), f"#{face}", font=mono, fill=(90, 90, 90))
            draw.text((590, y), facet_by_face[face].name, font=mono, fill=(15, 15, 15))
        else:
            draw.text((400, y), "不贴 / DO NOT APPLY", font=body, fill=(190, 60, 60))
            draw.text(
                (640, y),
                f"#{face}  该面有 R7.5 圆角缺口、图样却是直边，且解算不使用",
                font=body,
                fill=(150, 90, 90),
            )
        y += 30
    y += 8
    for anchor in sorted(model.anchors, key=lambda a: a.marker_id):
        draw.text((26, y), f"A{anchor.marker_id}", font=mono, fill=(15, 15, 15))
        draw.rectangle([100, y + 2, 124, y + 18], fill=(255, 255, 255), outline=(60, 60, 60))
        draw.text(
            (132, y), f"{anchor.dictionary} id {anchor.marker_id}", font=mono, fill=(90, 90, 90)
        )
        draw.text(
            (400, y),
            f"{view_of_face.get(int(anchor.face_entity_id), '-')}",
            font=mono,
            fill=(15, 15, 15),
        )
        draw.text((470, y), f"#{int(anchor.face_entity_id)}", font=mono, fill=(90, 90, 90))
        draw.text(
            (590, y),
            f"{anchor.name}   {anchor.marker_size_m * 1000:.0f} mm marker 居中于 {anchor.pad_size_m * 1000:.0f} mm pad",
            font=body,
            fill=(15, 15, 15),
        )
        y += 30
    y += 10
    notes = [
        "贴纸方向：每张图样的外轮廓就是裁切轮廓，也就是 CAD 面的轮廓——对齐轮廓即可，1.00 mm 黑边朝内。",
        "ArUco 贴纸的旋转（paste_quadrant）在描述符里仍是 null，装完后要实测一次并写回。",
    ]
    if mirror:
        notes[:0] = [
            "印张的彩色轮廓整体镜像了：每张贴纸实际吻合的是它孪生面（0.00 mm）。按本图的 decal 号贴，"
            "不要按印张上每张图样旁边写的 face 号。",
            "The sheet's coloured outlines are mirrored; each decal fits its twin face exactly.",
            "Follow the decal ids in this drawing, not the face numbers printed on the sheet.",
        ]
    for line in notes:
        draw.text((26, y), line, font=body, fill=(70, 70, 70))
        y += 27
    sheet.save(args.out)

    print(f"[OK] wrote {args.out} ({sheet.width}x{sheet.height})")
    print(
        f"[OK] {len(chosen)} views cover {len(label_of)} labelled faces: "
        + ", ".join(
            f"view {n} az {az:+.0f} el {el:+.0f} -> {len(assigned[(az, el)])}"
            for n, (az, el) in enumerate(chosen, start=1)
        )
    )
    for view_number, view in enumerate(chosen, start=1):
        for face in sorted(assigned[view], key=lambda f: label_of[f]):
            print(
                f"[OK] {label_of[face]:>4s} labelled in view {view_number}: {seen[view][face]:6d} px visible"
            )
    for decal, face in dead:
        print(
            f"[OK] {decal} (face #{face}) listed as not applied -- printed but not in the descriptor"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
