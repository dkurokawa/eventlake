"""Demo: reconstructing label history for an ML pipeline.

An application database that only keeps "the current label" for each item
loses the fact that a prediction was ever wrong, when it was corrected, and
by whom - the very history an ML team needs to measure model quality and
debug regressions.

eventlake keeps every PredictionMade and LabelCorrected event instead. This
script writes synthetic (randomly generated, seed-fixed - no external data)
events into a throwaway eventlake root and then asks it three questions a
"current label" table cannot answer:

  1. What fraction of predictions were later corrected by a human?
  2. How does that correction rate differ by model version?
  3. What did we believe the correct label was, as of a given point in time?
"""

from __future__ import annotations

import random
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import ClassVar

from eventlake.event import Event
from eventlake.lake import Lake
from eventlake.writer import Writer

LABELS = ("cat", "dog", "bird")
MODEL_VERSIONS = ("v1", "v2")
REVIEWERS = ("alice", "bob", "carol")

SEED = 20260927
NUM_ITEMS = 200
BASE_CORRECTION_RATE = 0.3

BASE_TIME = datetime(2026, 1, 1, tzinfo=UTC)


class PredictionMade(Event):
    event_type: ClassVar[str] = "prediction_made"

    item_id: str
    model_version: str
    predicted_label: str
    confidence: float


class LabelCorrected(Event):
    event_type: ClassVar[str] = "label_corrected"

    item_id: str
    old_label: str
    new_label: str
    reviewer: str


def generate_events(rng: random.Random) -> tuple[list[PredictionMade], list[LabelCorrected]]:
    """Synthesize a plausible ML labeling history.

    v2 is modeled as a better model: its predictions are corrected less
    often than v1's. All values (labels, timings, reviewers) are drawn from
    a seeded RNG - nothing here comes from a real dataset.
    """
    predictions: list[PredictionMade] = []
    corrections: list[LabelCorrected] = []

    for i in range(NUM_ITEMS):
        item_id = f"item-{i:04d}"
        model_version = rng.choice(MODEL_VERSIONS)
        predicted_label = rng.choice(LABELS)
        confidence = round(rng.uniform(0.5, 0.99), 3)
        occurred_at = BASE_TIME + timedelta(minutes=i)

        predictions.append(
            PredictionMade(
                item_id=item_id,
                model_version=model_version,
                predicted_label=predicted_label,
                confidence=confidence,
                occurred_at=occurred_at,
            )
        )

        correction_chance = BASE_CORRECTION_RATE * (0.6 if model_version == "v2" else 1.4)
        if rng.random() < correction_chance:
            other_labels = [label for label in LABELS if label != predicted_label]
            new_label = rng.choice(other_labels)
            corrected_at = occurred_at + timedelta(hours=rng.randint(1, 72))
            corrections.append(
                LabelCorrected(
                    item_id=item_id,
                    old_label=predicted_label,
                    new_label=new_label,
                    reviewer=rng.choice(REVIEWERS),
                    occurred_at=corrected_at,
                )
            )

    return predictions, corrections


def main() -> None:
    rng = random.Random(SEED)
    predictions, corrections = generate_events(rng)

    with tempfile.TemporaryDirectory(prefix="eventlake-example-") as tmp_dir:
        root = Path(tmp_dir)
        with Writer(root) as writer:
            writer.write_many(predictions)
            writer.write_many(corrections)

        lake = Lake(root)

        total_predictions = lake.sql("SELECT COUNT(*) FROM prediction_made").fetchone()
        total_corrections = lake.sql(
            "SELECT COUNT(DISTINCT item_id) FROM label_corrected"
        ).fetchone()
        assert total_predictions is not None
        assert total_corrections is not None
        n_predictions = total_predictions[0]
        n_corrected = total_corrections[0]
        overall_rate = n_corrected / n_predictions

        print("=== eventlake label history demo ===")
        print(f"predictions: {n_predictions}, later corrected: {n_corrected}")
        print(f"overall correction rate: {overall_rate:.1%}")
        print()

        print("correction rate by model version (a DB overwritten in place")
        print("could not tell you this - it never held the wrong label at all):")
        by_version = lake.sql(
            """
            SELECT
                p.model_version,
                COUNT(*) AS predictions,
                COUNT(c.item_id) AS corrected,
                COUNT(c.item_id) * 1.0 / COUNT(*) AS correction_rate
            FROM prediction_made p
            LEFT JOIN label_corrected c USING (item_id)
            GROUP BY p.model_version
            ORDER BY p.model_version
            """
        ).fetchall()
        for model_version, n_pred, n_corr, rate in by_version:
            print(f"  {model_version}: {n_corr}/{n_pred} corrected ({rate:.1%})")
        print()

        cutoff = BASE_TIME + timedelta(hours=24)
        as_of = lake.state_as_of("label_corrected", key="item_id", at=cutoff)
        known_corrections_count = len(as_of.fetchall())
        print(f"as of {cutoff.isoformat()}, {known_corrections_count} corrections were known")
        print("(a mutable 'current label' table cannot answer this at all - it only")
        print(" ever holds the latest value, with no record of when it changed)")


if __name__ == "__main__":
    main()
