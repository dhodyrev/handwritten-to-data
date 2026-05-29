"""Detection post-processing: NMS on duplicate line bboxes (Phase 1)."""
from __future__ import annotations

from .schemas import RegionTask


def _iou(a: list[int], b: list[int]) -> float:
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def nms_tasks(tasks: list[RegionTask], iou_thresh: float = 0.5) -> list[RegionTask]:
    """Drop near-duplicate line bboxes that hurt DetF1 precision AND inflate
    the concatenated Page CER text. We have no confidence score from Qwen, so
    "keep larger bbox" is the tie-breaker — it usually captures more of the
    line and yields a cleaner transcribe crop.

    Bboxes of different rtypes are kept independently — a formula box
    overlapping a handwritten text box is a real semantic distinction.
    """
    if not tasks:
        return tasks

    by_type: dict[str, list[tuple[int, RegionTask]]] = {}
    for i, t in enumerate(tasks):
        by_type.setdefault(t["rtype"], []).append((i, t))

    keep_idx: set[int] = set()
    for type_tasks in by_type.values():
        # sort by area descending — larger boxes win duplicate matches
        sorted_t = sorted(
            type_tasks,
            key=lambda it: -((it[1]["bbox"][2] - it[1]["bbox"][0])
                             * (it[1]["bbox"][3] - it[1]["bbox"][1])),
        )
        kept: list[tuple[int, RegionTask]] = []
        for idx, t in sorted_t:
            if all(_iou(t["bbox"], k[1]["bbox"]) < iou_thresh for k in kept):
                kept.append((idx, t))
        for idx, _ in kept:
            keep_idx.add(idx)

    return [tasks[i] for i in range(len(tasks)) if i in keep_idx]
