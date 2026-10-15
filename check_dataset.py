from __future__ import annotations

import argparse
import hashlib
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
EXPECTED_NAMES = {0: "fire", 1: "smoke"}


@dataclass
class SplitStats:
    images: int = 0
    labels: int = 0
    empty_labels: int = 0
    unreadable_images: int = 0
    boxes_by_class: dict[str, int] = field(
        default_factory=lambda: {name: 0 for name in EXPECTED_NAMES.values()}
    )


@dataclass
class DatasetAudit:
    data_yaml: Path
    dataset_root: Path
    split_stats: dict[str, SplitStats] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.errors


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as image_file:
        for chunk in iter(lambda: image_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _configured_names(raw_names: object) -> dict[int, str]:
    if isinstance(raw_names, list):
        return {index: str(name) for index, name in enumerate(raw_names)}
    if isinstance(raw_names, dict):
        try:
            return {int(index): str(name) for index, name in raw_names.items()}
        except (TypeError, ValueError):
            return {}
    return {}


def audit_dataset(data_yaml: Path) -> DatasetAudit:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "Thiếu PyYAML. Cài dependencies bằng: "
            "venv\\Scripts\\python.exe -m pip install -r requirements.txt"
        ) from exc

    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "Thiếu Pillow. Cài dependencies bằng: "
            "venv\\Scripts\\python.exe -m pip install -r requirements.txt"
        ) from exc

    config_path = data_yaml.resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy file cấu hình dataset: {config_path}")

    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8-sig"))
    except yaml.YAMLError as exc:
        raise ValueError(f"YAML không hợp lệ ({config_path}): {exc}") from exc
    if not isinstance(config, dict):
        raise ValueError(f"Nội dung YAML phải là mapping: {config_path}")

    configured_root = Path(str(config.get("path", "."))).expanduser()
    if not configured_root.is_absolute():
        configured_root = config_path.parent / configured_root
    configured_root = configured_root.resolve()

    report = DatasetAudit(config_path, configured_root)

    actual_names = _configured_names(config.get("names"))
    if actual_names != EXPECTED_NAMES:
        report.errors.append(
            "data.yaml phải giữ nguyên class ID: 0=fire, 1=smoke; "
            f"đang khai báo {actual_names!r}."
        )
    if config.get("nc", len(actual_names)) != len(EXPECTED_NAMES):
        report.errors.append(
            f"nc phải bằng {len(EXPECTED_NAMES)}, đang là {config.get('nc')!r}."
        )

    split_paths: dict[str, Path] = {}
    for split_name, yaml_key in (("train", "train"), ("valid", "val"), ("test", "test")):
        configured_path = config.get(yaml_key)
        if not isinstance(configured_path, str) or not configured_path.strip():
            report.errors.append(f"data.yaml thiếu đường dẫn hợp lệ cho split '{yaml_key}'.")
            continue
        image_dir = Path(configured_path).expanduser()
        if not image_dir.is_absolute():
            image_dir = configured_root / image_dir
        split_paths[split_name] = image_dir.resolve()

    try:
        image_root = configured_root.is_dir()
    except OSError:
        image_root = False
    if not image_root:
        report.errors.append(f"Không tìm thấy thư mục dataset: {configured_root}")

    image_hash_splits: dict[str, set[str]] = {}
    for split_name in ("train", "valid", "test"):
        stats = SplitStats()
        report.split_stats[split_name] = stats
        image_dir = split_paths.get(split_name)
        if image_dir is None:
            continue
        if not image_dir.is_dir():
            report.errors.append(
                f"Thiếu thư mục ảnh {split_name}: {image_dir}"
            )
            continue

        label_dir = image_dir.parent / "labels"
        if not label_dir.is_dir():
            report.errors.append(
                f"Thiếu thư mục nhãn {split_name}: {label_dir}"
            )
            continue

        try:
            images = sorted(
                path
                for path in image_dir.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            )
            labels = sorted(
                path
                for path in label_dir.iterdir()
                if path.is_file() and path.suffix.lower() == ".txt"
            )
        except OSError as exc:
            report.errors.append(f"Không đọc được split '{split_name}': {exc}")
            continue

        stats.images = len(images)
        stats.labels = len(labels)
        if not images:
            report.errors.append(f"Split '{split_name}' không có ảnh hợp lệ: {image_dir}")

        image_by_stem: dict[str, Path] = {}
        duplicate_image_stems: list[str] = []
        for image_path in images:
            stem = image_path.stem.casefold()
            if stem in image_by_stem:
                duplicate_image_stems.append(image_path.name)
            image_by_stem[stem] = image_path

        label_by_stem: dict[str, Path] = {}
        duplicate_label_stems: list[str] = []
        for label_path in labels:
            stem = label_path.stem.casefold()
            if stem in label_by_stem:
                duplicate_label_stems.append(label_path.name)
            label_by_stem[stem] = label_path

        if duplicate_image_stems:
            report.errors.append(
                f"{split_name}: có {len(duplicate_image_stems)} ảnh trùng tên stem "
                f"(ví dụ: {duplicate_image_stems[:5]})."
            )
        if duplicate_label_stems:
            report.errors.append(
                f"{split_name}: có {len(duplicate_label_stems)} file nhãn trùng tên stem "
                f"(ví dụ: {duplicate_label_stems[:5]})."
            )

        missing_labels = sorted(set(image_by_stem) - set(label_by_stem))
        orphan_labels = sorted(set(label_by_stem) - set(image_by_stem))
        if missing_labels:
            report.errors.append(
                f"{split_name}: {len(missing_labels)} ảnh thiếu nhãn "
                f"(ví dụ: {missing_labels[:5]})."
            )
        if orphan_labels:
            report.errors.append(
                f"{split_name}: {len(orphan_labels)} nhãn không có ảnh tương ứng "
                f"(ví dụ: {orphan_labels[:5]})."
            )

        unreadable_images: list[str] = []
        for image_path in images:
            try:
                with Image.open(image_path) as image:
                    image.verify()
            except (OSError, SyntaxError, ValueError):
                unreadable_images.append(image_path.name)
                continue

            try:
                image_hash = _sha256(image_path)
            except OSError as exc:
                report.errors.append(f"Không đọc được ảnh để kiểm tra trùng lặp {image_path}: {exc}")
                continue
            image_hash_splits.setdefault(image_hash, set()).add(split_name)

        stats.unreadable_images = len(unreadable_images)
        if unreadable_images:
            report.errors.append(
                f"{split_name}: {len(unreadable_images)} ảnh không đọc được "
                f"(ví dụ: {unreadable_images[:5]})."
            )

        malformed_labels: list[str] = []
        for label_path in labels:
            try:
                lines = label_path.read_text(encoding="utf-8-sig").splitlines()
            except (OSError, UnicodeError) as exc:
                malformed_labels.append(f"{label_path.name}: không đọc được ({exc})")
                continue

            nonempty_lines = [(line_number, line.strip()) for line_number, line in enumerate(lines, 1) if line.strip()]
            if not nonempty_lines:
                stats.empty_labels += 1
                continue

            for line_number, line in nonempty_lines:
                fields = line.split()
                location = f"{label_path.name}:{line_number}"
                try:
                    if len(fields) != 5:
                        raise ValueError(f"cần đúng 5 cột, nhận {len(fields)}")
                    class_id = int(fields[0])
                    if class_id not in EXPECTED_NAMES:
                        raise ValueError(f"class_id {class_id} không thuộc 0..1")
                    coordinates = [float(value) for value in fields[1:]]
                    if not all(math.isfinite(value) for value in coordinates):
                        raise ValueError("tọa độ không phải số hữu hạn")
                    if not all(0.0 <= value <= 1.0 for value in coordinates):
                        raise ValueError("tọa độ nằm ngoài [0,1]")
                    if coordinates[2] <= 0.0 or coordinates[3] <= 0.0:
                        raise ValueError("width và height phải lớn hơn 0")
                except (ValueError, OverflowError) as exc:
                    malformed_labels.append(f"{location}: {exc}")
                    continue
                stats.boxes_by_class[EXPECTED_NAMES[class_id]] += 1

        if malformed_labels:
            report.errors.append(
                f"{split_name}: {len(malformed_labels)} dòng nhãn sai định dạng "
                f"(ví dụ: {malformed_labels[:5]})."
            )

    leaked_hashes = [
        image_hash
        for image_hash, splits in image_hash_splits.items()
        if len(splits) > 1
    ]
    if leaked_hashes:
        report.errors.append(
            f"Có {len(leaked_hashes)} ảnh trùng nội dung giữa các split; "
            "cần xử lý để tránh rò rỉ dữ liệu."
        )

    return report


def format_report(report: DatasetAudit) -> str:
    lines = [
        f"Dataset YAML: {report.data_yaml}",
        f"Dataset root: {report.dataset_root}",
        "Class mapping: 0=fire, 1=smoke",
    ]
    for split_name in ("train", "valid", "test"):
        stats = report.split_stats.get(split_name, SplitStats())
        class_counts = ", ".join(
            f"{name}={count}" for name, count in stats.boxes_by_class.items()
        )
        lines.append(
            f"{split_name}: images={stats.images}, labels={stats.labels}, "
            f"empty_labels={stats.empty_labels}, unreadable={stats.unreadable_images}, "
            f"boxes[{class_counts}]"
        )
    if report.errors:
        lines.append(f"Dataset validation FAILED ({len(report.errors)} issue(s)):")
        lines.extend(f"- {error}" for error in report.errors)
    else:
        lines.append("Dataset validation PASSED.")
        lines.append(
            "File nhãn rỗng được tính là ảnh âm tính (không có đối tượng), không phải lỗi."
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Kiểm tra cấu hình, ảnh, nhãn YOLO và trùng lặp giữa các split."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).resolve().parent / "dataset" / "data.yaml",
        help="Đường dẫn tới data.yaml (mặc định: dataset/data.yaml).",
    )
    args = parser.parse_args()

    try:
        report = audit_dataset(args.data)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Lỗi kiểm tra dataset: {exc}", file=sys.stderr)
        return 1

    print(format_report(report))
    return 0 if report.is_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
