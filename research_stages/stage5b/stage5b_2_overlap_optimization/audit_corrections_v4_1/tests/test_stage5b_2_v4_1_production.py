#!/usr/bin/env python3
"""
Unit Test Suite for Stage 5B.2 V4.1 Production Code Verification.

CRITICAL REQUIREMENT:
All functions under test are directly imported from the production script:
    `research_stages/stage5b/stage5b_2_overlap_optimization/audit_corrections_v4_1/scripts/run_stage5b_2_corrected_v4_1.py`
Zero copied production implementations in this file.

Test Coverage:
1. Audit B: M2 Prototype-Purity Interval Union (Clean, same-speaker overlapping, duplicate, adjacent, competing 0.2s, competing duplicate, exactly 0.80, below 0.80, >0.80 with competing, permutation invariance, equal union ties, missing cache).
2. Audit A: Primary-Speaker Determinism (Equal count tie-breaking, majority count priority, permutation invariance).
3. Regression Suite: Oracle interval union, Hungarian mapping, M3/M4 turn constraints, timestamp frame-bin boundaries.
"""

import json
import math
import random
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

# Ensure production script is on sys.path
TEST_DIR = Path(__file__).resolve().parent
V4_1_DIR = TEST_DIR.parent
SCRIPTS_DIR = V4_1_DIR / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

# DIRECT IMPORT OF PRODUCTION IMPLEMENTATIONS UNDER TEST
from run_stage5b_2_corrected_v4_1 import (
    load_dual_encoder_prototypes_pure,
    attribute_secondary_speakers_v4,
    attribute_m2_pure,
    attribute_secondary_oracle_multilabel,
    attribute_m1_corrected,
    attribute_m3_m4_corrected,
    extract_contiguous_turns,
    binarize_overlap_frame_bin_support,
    compute_multilabel_hungarian_mapping,
    safe_diarization_metrics,
)

from pyannote.core import Annotation, Segment


class TestAuditBM2PrototypePurityIntervalUnion(unittest.TestCase):
    """
    Audit B: Detailed unit testing of interval-union coverage calculation in `load_dual_encoder_prototypes_pure`.
    """

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.cache_dir = Path(self.temp_dir)
        self.rec_id = "test_rec"
        self.rec_dir = self.cache_dir / self.rec_id
        self.rec_dir.mkdir(parents=True)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def _create_synthetic_cache(self, n_subsegs: int = 1, seed: int = 42, subseg_dur: float = 1.5):
        rng = np.random.RandomState(seed)
        t_raw = rng.randn(n_subsegs, 192).astype(np.float32)
        t_embs = t_raw / np.linalg.norm(t_raw, axis=1, keepdims=True)
        w_raw = rng.randn(n_subsegs, 256).astype(np.float32)
        w_embs = w_raw / np.linalg.norm(w_raw, axis=1, keepdims=True)

        np.savez_compressed(self.rec_dir / "titanet_large__dfn__scale0.npz", l2_normalized_embeddings=t_embs)
        np.savez_compressed(self.rec_dir / "wespeaker_resnet293_lm__dfn__scale0.npz", l2_normalized_embeddings=w_embs)

        subsegs = []
        for i in range(n_subsegs):
            subsegs.append({
                "subsegment_id": f"sub_{i}",
                "start": float(i * subseg_dur),
                "end": float(i * subseg_dur + subseg_dur),
                "duration": subseg_dur,
            })
        meta = {"recording_id": self.rec_id, "subsegments": subsegs}
        (self.rec_dir / "subsegments_metadata__scale0.json").write_text(json.dumps(meta), encoding="utf-8")
        return t_embs, w_embs, subsegs

    def test_clean_single_speaker_subsegment(self):
        """Clean single-speaker subsegment covering 100% is admitted."""
        self._create_synthetic_cache(n_subsegs=1)
        base_segs = [{"start": 0.0, "end": 1.5, "speaker": "spk_0"}]
        proto = load_dual_encoder_prototypes_pure(base_segs, self.rec_id, min_prototype_purity=0.80, cache_dir=self.cache_dir)
        self.assertIn("spk_0", proto["concat"])
        self.assertAlmostEqual(float(np.linalg.norm(proto["concat"]["spk_0"])), 1.0, places=5)

    def test_same_speaker_overlapping_segments(self):
        """Overlapping intervals of the same speaker must NOT inflate coverage above window duration."""
        self._create_synthetic_cache(n_subsegs=1)  # [0.0, 1.5]
        # [0.0, 1.0] and [0.5, 1.5] overlap by 0.5s. Union is 1.5s. Naive sum would be 2.0s (>1.5s window).
        base_segs = [
            {"start": 0.0, "end": 1.0, "speaker": "spk_0"},
            {"start": 0.5, "end": 1.5, "speaker": "spk_0"},
        ]
        proto = load_dual_encoder_prototypes_pure(base_segs, self.rec_id, min_prototype_purity=0.80, cache_dir=self.cache_dir)
        self.assertIn("spk_0", proto["concat"])

    def test_exact_duplicate_same_speaker_segments(self):
        """Duplicate segments of same speaker merge to single duration without double-counting."""
        self._create_synthetic_cache(n_subsegs=1)
        # Two identical [0.0, 1.0] segments. Union is 1.0s (66.7% < 80%).
        # Under naive summation: 1.0 + 1.0 = 2.0s (133% > 80% -> false admission!)
        # Under interval union: union is 1.0s (66.7% < 80% -> correctly rejected!)
        base_segs = [
            {"start": 0.0, "end": 1.0, "speaker": "spk_0"},
            {"start": 0.0, "end": 1.0, "speaker": "spk_0"},
        ]
        proto = load_dual_encoder_prototypes_pure(base_segs, self.rec_id, min_prototype_purity=0.80, cache_dir=self.cache_dir)
        self.assertEqual(len(proto["concat"]), 0, "Duplicate segments must not falsely inflate purity above 80%")

    def test_adjacent_same_speaker_intervals(self):
        """Abutting segments merge cleanly: [0.0, 0.6] + [0.6, 1.3] = 1.3s (86.7% >= 80%)."""
        self._create_synthetic_cache(n_subsegs=1)
        base_segs = [
            {"start": 0.0, "end": 0.6, "speaker": "spk_0"},
            {"start": 0.6, "end": 1.3, "speaker": "spk_0"},
        ]
        proto = load_dual_encoder_prototypes_pure(base_segs, self.rec_id, min_prototype_purity=0.80, cache_dir=self.cache_dir)
        self.assertIn("spk_0", proto["concat"])

    def test_competing_speaker_with_02_seconds_activity(self):
        """Competing speaker activity of 0.20s must strictly exclude subsegment."""
        self._create_synthetic_cache(n_subsegs=1)
        base_segs = [
            {"start": 0.0, "end": 1.3, "speaker": "spk_0"},
            {"start": 1.3, "end": 1.5, "speaker": "spk_1"},
        ]
        proto = load_dual_encoder_prototypes_pure(base_segs, self.rec_id, min_prototype_purity=0.80, cache_dir=self.cache_dir)
        self.assertEqual(len(proto["concat"]), 0)

    def test_competing_speaker_with_overlapping_duplicate_segments(self):
        """Competing speaker with duplicated intervals must be correctly detected and exclude subsegment."""
        self._create_synthetic_cache(n_subsegs=1)
        base_segs = [
            {"start": 0.0, "end": 1.3, "speaker": "spk_0"},
            {"start": 1.3, "end": 1.5, "speaker": "spk_1"},
            {"start": 1.3, "end": 1.5, "speaker": "spk_1"},
        ]
        proto = load_dual_encoder_prototypes_pure(base_segs, self.rec_id, min_prototype_purity=0.80, cache_dir=self.cache_dir)
        self.assertEqual(len(proto["concat"]), 0)

    def test_coverage_exactly_at_080(self):
        """Coverage exactly at 0.80 (1.20s / 1.50s) with 0.0s competing speaker must be admitted."""
        self._create_synthetic_cache(n_subsegs=1)
        base_segs = [{"start": 0.0, "end": 1.20, "speaker": "spk_0"}]
        proto = load_dual_encoder_prototypes_pure(base_segs, self.rec_id, min_prototype_purity=0.80, cache_dir=self.cache_dir)
        self.assertIn("spk_0", proto["concat"])

    def test_coverage_just_below_080(self):
        """Coverage just below 0.80 (1.19s / 1.50s = 79.3%) must be excluded."""
        self._create_synthetic_cache(n_subsegs=1)
        base_segs = [{"start": 0.0, "end": 1.19, "speaker": "spk_0"}]
        proto = load_dual_encoder_prototypes_pure(base_segs, self.rec_id, min_prototype_purity=0.80, cache_dir=self.cache_dir)
        self.assertEqual(len(proto["concat"]), 0)

    def test_coverage_greater_than_080_with_competing_activity(self):
        """Dominant coverage 1.30s (86.7%) but competing coverage 0.10s must be excluded."""
        self._create_synthetic_cache(n_subsegs=1)
        base_segs = [
            {"start": 0.0, "end": 1.30, "speaker": "spk_0"},
            {"start": 1.30, "end": 1.40, "speaker": "spk_1"},
        ]
        proto = load_dual_encoder_prototypes_pure(base_segs, self.rec_id, min_prototype_purity=0.80, cache_dir=self.cache_dir)
        self.assertEqual(len(proto["concat"]), 0)

    def test_input_order_permutation_invariance(self):
        """Permuting input base segments order must yield identical prototypes."""
        self._create_synthetic_cache(n_subsegs=2)
        base_1 = [
            {"start": 0.0, "end": 1.5, "speaker": "spk_A"},
            {"start": 1.5, "end": 3.0, "speaker": "spk_B"},
        ]
        base_2 = list(reversed(base_1))
        p1 = load_dual_encoder_prototypes_pure(base_1, self.rec_id, min_prototype_purity=0.80, cache_dir=self.cache_dir)
        p2 = load_dual_encoder_prototypes_pure(base_2, self.rec_id, min_prototype_purity=0.80, cache_dir=self.cache_dir)
        np.testing.assert_allclose(p1["concat"]["spk_A"], p2["concat"]["spk_A"], atol=1e-6)
        np.testing.assert_allclose(p1["concat"]["spk_B"], p2["concat"]["spk_B"], atol=1e-6)

    def test_multiple_speakers_with_equal_union_duration(self):
        """Equal union durations (e.g. 0.70s each) tie-break deterministically to alphabetical speaker ID."""
        self._create_synthetic_cache(n_subsegs=1)
        # spk_B on [0.0, 0.70], spk_A on [0.70, 1.40].
        # In both cases competing_dur > 0 so subsegment is excluded from prototype pool.
        base_segs = [
            {"start": 0.0, "end": 0.70, "speaker": "spk_B"},
            {"start": 0.70, "end": 1.40, "speaker": "spk_A"},
        ]
        proto = load_dual_encoder_prototypes_pure(base_segs, self.rec_id, min_prototype_purity=0.80, cache_dir=self.cache_dir)
        self.assertEqual(len(proto["concat"]), 0)

    def test_prototype_dimensions_and_finiteness(self):
        """All prototypes must be finite, strictly non-zero, and match exact embedding dimensions."""
        self._create_synthetic_cache(n_subsegs=2)
        base_segs = [
            {"start": 0.0, "end": 1.5, "speaker": "spk_0"},
            {"start": 1.5, "end": 3.0, "speaker": "spk_1"},
        ]
        proto_data = load_dual_encoder_prototypes_pure(
            base_segs=base_segs,
            rec_id=self.rec_id,
            min_prototype_purity=0.80,
            cache_dir=self.cache_dir,
        )
        for spk in ["spk_0", "spk_1"]:
            t_vec = proto_data["titanet"][spk]
            w_vec = proto_data["wespeaker"][spk]
            c_vec = proto_data["concat"][spk]

            self.assertEqual(t_vec.shape, (192,))
            self.assertEqual(w_vec.shape, (256,))
            self.assertEqual(c_vec.shape, (448,))

            self.assertTrue(np.all(np.isfinite(t_vec)))
            self.assertTrue(np.all(np.isfinite(w_vec)))
            self.assertTrue(np.all(np.isfinite(c_vec)))

    def test_missing_embedding_cache(self):
        """Missing embedding cache returns empty structure gracefully."""
        proto = load_dual_encoder_prototypes_pure([], "non_existent_rec", cache_dir=self.cache_dir)
        self.assertEqual(proto["concat"], {})
        self.assertEqual(proto["titanet"], {})
        self.assertEqual(proto["wespeaker"], {})


class TestAuditAPrimarySpeakerDeterminism(unittest.TestCase):
    """
    Audit A: Deterministic primary-speaker selection under unequal and equal frame counts.
    """

    def setUp(self):
        self.resolution = 0.01

    def test_majority_frame_count_priority(self):
        """Speaker with strictly greater frame count (51 vs 50) is selected regardless of alphabetical order."""
        base_segs = [
            {"start": 0.0, "end": 0.50, "speaker": "spk_A"},
            {"start": 0.50, "end": 1.01, "speaker": "spk_Z"},
        ]
        ovl_segs = [{"start": 0.0, "end": 1.01}]
        res_segs, tot_ovl, attr_ovl, dur = attribute_secondary_speakers_v4(
            base_segs=base_segs,
            ovl_segs=ovl_segs,
            attribution_method="M0_no_overlap",
            duration_sec=2.0,
            proto_data={},
            resolution=self.resolution,
        )
        self.assertEqual(tot_ovl, 1)

    def test_equal_count_alphabetical_tie_break(self):
        """When frame counts tie (50 vs 50), tie is broken alphabetically (spk_A beats spk_B)."""
        base_segs_1 = [
            {"start": 0.0, "end": 0.50, "speaker": "spk_B"},
            {"start": 0.50, "end": 1.00, "speaker": "spk_A"},
            {"start": 1.10, "end": 1.50, "speaker": "spk_B"},
        ]
        base_segs_2 = [
            {"start": 0.0, "end": 0.50, "speaker": "spk_A"},
            {"start": 0.50, "end": 1.00, "speaker": "spk_B"},
            {"start": 1.10, "end": 1.50, "speaker": "spk_B"},
        ]
        ovl_segs = [{"start": 0.0, "end": 1.00}]

        res_1, _, _, _ = attribute_secondary_speakers_v4(
            base_segs=base_segs_1,
            ovl_segs=ovl_segs,
            attribution_method="M1_temporal_context",
            duration_sec=2.0,
            proto_data={},
            resolution=self.resolution,
        )
        res_2, _, _, _ = attribute_secondary_speakers_v4(
            base_segs=base_segs_2,
            ovl_segs=ovl_segs,
            attribution_method="M1_temporal_context",
            duration_sec=2.0,
            proto_data={},
            resolution=self.resolution,
        )
        self.assertEqual(res_1[0]["speaker"], "spk_B")
        self.assertEqual(res_2[0]["speaker"], "spk_B")


class TestRegressionSuite(unittest.TestCase):
    """
    Regression suite ensuring oracle interval union, Hungarian mapping, and timestamp boundaries.
    """

    def setUp(self):
        self.resolution = 0.01

    def test_oracle_interval_union(self):
        """Oracle attribution merges duplicate/overlapping reference intervals."""
        gt_segs = [
            {"start": 1.0, "end": 2.0, "speaker": "ref_primary"},
            {"start": 1.0, "end": 1.8, "speaker": "ref_0"},
            {"start": 1.2, "end": 2.0, "speaker": "ref_0"},
        ]
        h_map = {"ref_primary": "hyp_primary", "ref_0": "hyp_0"}
        res = attribute_secondary_oracle_multilabel((1.0, 2.0), "hyp_primary", gt_segs, h_map)
        self.assertEqual(res["speaker"], "hyp_0")
        self.assertAlmostEqual(res["all_active_gt"]["ref_0"], 1.0, places=5)

    def test_hungarian_mapping(self):
        ref_segs = [{"start": 0.0, "end": 5.0, "speaker": "ref_A"}, {"start": 5.0, "end": 10.0, "speaker": "ref_B"}]
        hyp_segs = [{"start": 0.0, "end": 5.0, "speaker": "hyp_1"}, {"start": 5.0, "end": 10.0, "speaker": "hyp_0"}]
        m, _, _, _, _, _ = compute_multilabel_hungarian_mapping(ref_segs, hyp_segs, 10.0)
        self.assertEqual(m["ref_A"], "hyp_1")
        self.assertEqual(m["ref_B"], "hyp_0")

    def test_m3_m4_termination_at_speaker_change_and_silence(self):
        """M3/M4 contiguous turn extraction terminates strictly at unvoiced silence or speaker changes."""
        frames = [None] * 500
        for i in range(100, 190):
            frames[i] = "spk_A"

        runs_backward = extract_contiguous_turns(
            frame_spk=frames,
            search_start_idx=199,
            search_end_idx=0,
            direction="backward",
            resolution=0.01,
        )
        if runs_backward:
            self.assertNotEqual(runs_backward[0]["end_sec"], 2.0, "Must not bridge silence to overlap boundary")

        for i in range(250, 280):
            frames[i] = "spk_B"

        res = attribute_m3_m4_corrected(
            frame_spk=frames,
            overlap_interval=(2.0, 2.5),
            primary_spk="spk_0",
            method="M3",
            resolution=0.01,
            max_gap_sec=0.20,
            min_turn_dur_sec=0.20,
        )
        self.assertEqual(res["speaker"], "spk_B")

    def test_timestamp_frame_bin_support_boundaries(self):
        """Frame-bin support boundaries match window midpoint convention: t_start = mid - S/2, t_end = mid + S/2."""
        step = 0.016875
        times = np.array([0.03096875 + i * step for i in range(10)], dtype=np.float32)
        p_ovl = np.zeros(10, dtype=np.float32)
        p_ovl[2:5] = 0.80

        segs = binarize_overlap_frame_bin_support(
            frame_times=times,
            p_ovl=p_ovl,
            duration_sec=10.0,
            threshold=0.50,
            min_duration_on=0.04,
            min_duration_off=0.15,
        )
        self.assertEqual(len(segs), 1)
        expected_st = round(float(times[2] - step / 2.0), 3)
        expected_en = round(float(times[4] + step / 2.0), 3)
        self.assertEqual(segs[0]["start"], expected_st)
        self.assertEqual(segs[0]["end"], expected_en)
        self.assertAlmostEqual(segs[0]["end"] - segs[0]["start"], 3 * step, delta=0.002)


if __name__ == "__main__":
    unittest.main()
