#!/usr/bin/env python3
"""
선을 넘어가는 차량 세기.
한 프레임의 박스만 세면 같은 차를 여러 번 센다. 추적 id를 붙인 뒤,
차가 선을 지날 때만 더한다. 결과는 mp4로 저장하고, --show이면 cv2.imshow로 본다.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
WORK = ROOT / "runs"
EXAMPLES_DIR = ROOT / "examples"
VEHICLE = ("car", "truck", "bus", "motorcycle")
EXAMPLES = {
    "vehicles": {
        "source": "supervision",
        "asset": "VEHICLES",
        "line": "horizontal",
        "line_y": 0.70,
        "threshold": 0.5,
        "blurb": "기존 쿡북 고속도로.",
    },
    "a40": {
        "source": "url",
        "url": "https://commons.wikimedia.org/wiki/Special:FilePath/Bochum_-_A40_(Br%C3%BCcke_L%C3%BCtkendorpweg)_01_(1)_ies.ogv",
        "filename": "a40.mp4",
        "convert_mp4": True,
        "line": "horizontal",
        "line_y": 0.58,
        "line_x0": 0.08,
        "line_x1": 0.92,
        "threshold": 0.35,
        "minimum_consecutive_frames": 3,
        "blurb": "보훔 A40. 고가에서 내려다본 양방향 아우토반.",
    },
    "a43": {
        "source": "url",
        "url": "https://commons.wikimedia.org/wiki/Special:FilePath/Bochum_-_A43_(Autobahnbr%C3%BCcke_In_der_Grume)_01_(1)_ies.ogv",
        "filename": "a43.mp4",
        "convert_mp4": True,
        "line": "horizontal",
        "line_y": 0.62,
        "line_x0": 0.04,
        "line_x1": 0.96,
        "threshold": 0.35,
        "blurb": "보훔 A43. 고가에서 내려다본 양방향 아우토반. 가까운 차가 크게 보인다.",
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="선을 넘는 차량을 세고 mp4로 저장한다")
    p.add_argument(
        "--example",
        choices=(*EXAMPLES, "all"),
        default="vehicles",
        help="강의용 예시. vehicles는 기존 고속도로, a40·a43은 보훔 아우토반",
    )
    p.add_argument("--video", type=Path, default=None, help="직접 넣은 영상. 있으면 --example보다 우선한다")
    p.add_argument("--width", type=int, default=1280, help="표시·추론 너비. 1080은 1280을 유지한다")
    p.add_argument("--height", type=int, default=720, help="표시·추론 높이. 1080은 720을 유지한다")
    p.add_argument(
        "--line-y",
        type=float,
        default=None,
        help="가로선 위치. 0은 위, 1은 아래. 가로선 예시에만 쓴다. 비우면 예시값을 쓴다",
    )
    p.add_argument("--max-frames", type=int, default=0, help="0이면 영상 끝까지")
    p.add_argument("--out", type=Path, default=None, help="비우면 runs/{예시이름}.mp4")
    p.add_argument(
        "--show",
        action="store_true",
        default=True,
        help="cv2.imshow 창을 연다. 화면이 없으면 저장만 한다",
    )
    return p.parse_args()


def fetch_url(url: str, filename: str, convert_mp4: bool = False) -> Path:
    dest = EXAMPLES_DIR / filename
    EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 1_000_000:
        return dest.resolve()
    import subprocess
    import urllib.request

    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
        "Referer": "https://www.pexels.com/",
    }
    print("받는 중", url)
    raw = dest if not convert_mp4 else dest.with_suffix(".download")
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=180) as src, open(raw, "wb") as out:
        out.write(src.read())
    if convert_mp4:
        subprocess.check_call(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(raw),
                "-an",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(dest),
            ]
        )
        raw.unlink(missing_ok=True)
    return dest.resolve()


def fetch_example(name: str) -> Path:
    from supervision.assets import VideoAssets, download_assets

    spec = EXAMPLES[name]
    if spec["source"] == "url":
        return fetch_url(spec["url"], spec["filename"], convert_mp4=bool(spec.get("convert_mp4")))
    EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    asset = getattr(VideoAssets, spec["asset"])
    return Path(download_assets(asset, directory=EXAMPLES_DIR)).resolve()


def source_video(user: Path | None, example: str) -> Path:
    if user is not None:
        path = user.expanduser().resolve()
        if not path.is_file():
            raise SystemExit(f"영상을 찾지 못했다: {path}")
        return path
    return fetch_example(example)


def crop_frame(frame, crop):
    if not crop:
        return frame
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = crop
    return frame[int(h * y0) : int(h * y1), int(w * x0) : int(w * x1)]


def make_line_zone(sv, Position, spec: dict, width: int, height: int):
    """선은 차 중심으로만 센다. 가로선은 오른쪽에서 왼쪽으로 두어 왼쪽 차로가 IN, 오른쪽 차로가 OUT이 된다."""
    anchors = (Position.CENTER,)
    if spec.get("line") == "vertical":
        x = int(width * spec["line_x"])
        y0 = int(height * spec["line_y0"])
        y1 = int(height * spec["line_y1"])
        print("선 x =", x, "y", y0, y1)
        return sv.LineZone(
            start=sv.Point(x, y0),
            end=sv.Point(x, y1),
            triggering_anchors=anchors,
        )
    y = int(height * spec["line_y"])
    x0 = int(width * spec.get("line_x0", 0.0))
    x1 = int(width * spec.get("line_x1", 1.0))
    print("선 y =", y, "x", x1, x0)
    return sv.LineZone(
        start=sv.Point(x1, y),
        end=sv.Point(x0, y),
        triggering_anchors=anchors,
    )


def detect_vehicles(model, frame, threshold: float = 0.5, min_area: float = 0.0):
    rgb = np.ascontiguousarray(frame[:, :, ::-1])
    detections = model.predict(rgb, include_source_image=False, threshold=threshold)
    if len(detections) == 0:
        return detections
    names = np.array(detections["class_name"])
    keep = np.isin(names, VEHICLE)
    detections = detections[keep]
    if min_area <= 0 or len(detections) == 0:
        return detections
    xyxy = detections.xyxy
    area = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    return detections[area >= min_area]


def class_counts(line_zone):
    in_counts = {name: 0 for name in VEHICLE}
    out_counts = {name: 0 for name in VEHICLE}
    names = {int(k): v for k, v in line_zone.class_id_to_name.items() if k is not None}
    for class_id, n in line_zone.in_count_per_class.items():
        if class_id is None:
            continue
        name = names.get(int(class_id))
        if name in in_counts:
            in_counts[name] += n
    for class_id, n in line_zone.out_count_per_class.items():
        if class_id is None:
            continue
        name = names.get(int(class_id))
        if name in out_counts:
            out_counts[name] += n
    return in_counts, out_counts


def class_banner_height(width: int) -> int:
    return 72 if width < 800 else 92


def draw_class_panel(frame, line_zone):
    """차종별 대수를 영상 아래 막대에 그린다. 왼쪽 IN, 오른쪽 OUT."""
    in_counts, out_counts = class_counts(line_zone)
    height, width = frame.shape[:2]
    narrow = width < 800
    banner_h = class_banner_height(width)
    banner = np.zeros((banner_h, width, 3), dtype=np.uint8)
    mid = width // 2
    banner[:, :mid] = (28, 42, 28)
    banner[:, mid:] = (48, 32, 24)
    cv2.line(banner, (mid, 8), (mid, banner_h - 8), (90, 90, 90), 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    title_scale = 0.62 if narrow else 0.82
    body_scale = 0.42 if narrow else 0.58
    title_th = 2
    body_th = 1 if narrow else 2
    pad = 12 if narrow else 20
    title_y = 26 if narrow else 30
    row0 = 48 if narrow else 58
    row1 = 66 if narrow else 82
    col_w = max(90, (mid - pad * 2) // 2)

    def put(text, x, y, scale, color, thickness):
        cv2.putText(banner, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)

    put(f"IN  {sum(in_counts.values())}", pad, title_y, title_scale, (170, 255, 170), title_th)
    put(f"OUT  {sum(out_counts.values())}", mid + pad, title_y, title_scale, (190, 215, 255), title_th)
    for i, name in enumerate(VEHICLE):
        col = i % 2
        y = row0 if i < 2 else row1
        put(
            f"{name}  {in_counts[name]}",
            pad + col * col_w,
            y,
            body_scale,
            (240, 240, 240),
            body_th,
        )
        put(
            f"{name}  {out_counts[name]}",
            mid + pad + col * col_w,
            y,
            body_scale,
            (240, 240, 240),
            body_th,
        )
    return np.vstack([frame, banner])


def annotate(frame, detections, line_zone, box_annotator, label_annotator, trace_annotator, line_annotator):
    names = detections["class_name"] if len(detections) else []
    if names is None:
        names = [""] * len(detections)
    labels = [
        f"#{tid} {name} {conf:.2f}"
        for tid, name, conf in zip(
            detections.tracker_id,
            names,
            detections.confidence if len(detections) else [],
        )
    ]
    out = frame.copy()
    out = trace_annotator.annotate(out, detections)
    out = box_annotator.annotate(out, detections)
    out = label_annotator.annotate(out, detections, labels)
    line_zone.trigger(detections)
    out = line_annotator.annotate(out, line_counter=line_zone)
    return draw_class_panel(out, line_zone)


def open_writer(path: Path, width: int, height: int, fps: float) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    fps = fps if fps and fps > 1 else 25.0
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise SystemExit(f"영상을 저장할 수 없다: {path}")
    return writer


def process_video(args, video_path: Path, out_path: Path, spec: dict, model, sv, Position) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"영상을 열지 못했다: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 1 else 25.0
    width = int(spec.get("width", args.width))
    height = int(spec.get("height", args.height))
    crop = spec.get("crop")

    tracker = sv.ByteTrack(
        frame_rate=fps,
        lost_track_buffer=int(spec.get("lost_track_buffer", 60)),
        minimum_matching_threshold=float(spec.get("minimum_matching_threshold", 0.5)),
        minimum_consecutive_frames=int(spec.get("minimum_consecutive_frames", 1)),
        track_activation_threshold=float(spec.get("track_activation_threshold", 0.25)),
    )
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.5)
    trace_annotator = sv.TraceAnnotator(thickness=2)
    line_annotator = sv.LineZoneAnnotator(
        thickness=3,
        text_thickness=1,
        text_scale=0.8,
        display_in_count=False,
        display_out_count=False,
    )
    writer = open_writer(out_path, width, height + class_banner_height(width), fps)

    show = args.show and bool(os.environ.get("DISPLAY"))
    if args.show and not show:
        print("DISPLAY가 없어 cv2.imshow는 건너뛰고 파일만 저장한다.")
    if show:
        print("창에서 q를 누르면 멈춘다.")

    line_zone = None
    index = 0
    print("입력", video_path)
    print("선", spec.get("line", "horizontal"), "threshold", spec["threshold"])
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        frame = crop_frame(frame, crop)
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        if line_zone is None:
            line_zone = make_line_zone(sv, Position, spec, width, height)

        detections = detect_vehicles(
            model,
            frame,
            threshold=spec["threshold"],
            min_area=float(spec.get("min_area", 0)),
        )
        detections = tracker.update_with_detections(detections)
        if detections.tracker_id is None:
            detections.tracker_id = np.zeros(len(detections), dtype=int)
        drawn = annotate(
            frame,
            detections,
            line_zone,
            box_annotator,
            label_annotator,
            trace_annotator,
            line_annotator,
        )
        writer.write(drawn)
        if show:
            cv2.imshow("line_count", drawn)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
        if index % 25 == 0:
            in_counts, out_counts = class_counts(line_zone)
            print(f"frame {index:03d}  in {in_counts}  out {out_counts}")
        index += 1
        if args.max_frames and index >= args.max_frames:
            break

    cap.release()
    writer.release()
    if show:
        cv2.destroyAllWindows()

    in_counts, out_counts = ({name: 0 for name in VEHICLE}, {name: 0 for name in VEHICLE})
    if line_zone is not None:
        in_counts, out_counts = class_counts(line_zone)
    print("frames", index)
    print("in", in_counts)
    print("out", out_counts)
    print("저장", out_path)


def jobs_from_args(args):
    if args.video is not None:
        spec = {
            "line": "horizontal",
            "line_y": 0.70 if args.line_y is None else args.line_y,
            "threshold": 0.5,
        }
        out = args.out if args.out is not None else WORK / "custom.mp4"
        return [("custom", source_video(args.video, "vehicles"), out.expanduser().resolve(), spec)]
    names = list(EXAMPLES) if args.example == "all" else [args.example]
    jobs = []
    for name in names:
        spec = dict(EXAMPLES[name])
        if spec.get("line", "horizontal") == "horizontal" and args.line_y is not None:
            spec["line_y"] = args.line_y
        out = WORK / f"{name}.mp4" if args.example == "all" or args.out is None else args.out
        jobs.append((name, fetch_example(name), Path(out).expanduser().resolve(), spec))
    return jobs


def main() -> None:
    args = parse_args()

    import torch
    import supervision as sv
    from rfdetr import RFDETRSmall
    from supervision.geometry.core import Position

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("torch", torch.__version__)
    print("device", device)
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        print("gpu", torch.cuda.get_device_name(0))
        print("vram_gb", round(props.total_memory / 1024**3, 1))

    model = RFDETRSmall()
    for name, video_path, out_path, spec in jobs_from_args(args):
        if name in EXAMPLES:
            print("예시", name, EXAMPLES[name]["blurb"])
        process_video(args, video_path, out_path, spec, model, sv, Position)


if __name__ == "__main__":
    main()
