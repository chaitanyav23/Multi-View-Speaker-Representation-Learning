#!/usr/bin/env python3
"""
Stage 5B.4-A.1.3 Remediation: Evaluation Adapters with Verified Phase 20 Parity
Integrates dual-collar DER/JER from pyannote.metrics and audited Phase 20 cpWER Hungarian matching.
"""

from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

from path_utils import get_stage5b_4_dir
A1_SCRIPTS = get_stage5b_4_dir() / "a1_remediation" / "scripts"
if str(A1_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(A1_SCRIPTS))

from evaluation_adapters import BaseEvaluationAdapter, SyntheticEvaluationAdapter
from cpwer_adapter import evaluate_cpwer


class ProductionEvaluationAdapterRemediated(BaseEvaluationAdapter):
    """
    Production evaluation adapter utilizing pyannote.metrics for dual-collar DER/JER
    and Phase 20 cpWER Hungarian matching with canonical {"segments": [...]} envelope.
    """

    def __init__(self):
        self.is_synthetic = False

    def evaluate(
        self,
        hypothesis_rttm: Path,
        reference_rttm: Path,
        aligned_words_json: Optional[Path] = None,
        reference_transcript_json: Optional[Path] = None,
        k_hat: Optional[int] = None,
        k_true: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        hyp_path = Path(hypothesis_rttm)
        ref_path = Path(reference_rttm)

        if not hyp_path.exists():
            raise FileNotFoundError(f"Production hypothesis RTTM not found: {hyp_path}")
        if not ref_path.exists():
            raise FileNotFoundError(f"Production reference RTTM not found: {ref_path}")

        from pyannote.core import Annotation, Segment
        from pyannote.metrics.diarization import DiarizationErrorRate, JaccardErrorRate

        def load_annotation(path: Path) -> Annotation:
            ann = Annotation()
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 8 and parts[0] == "SPEAKER":
                    st = float(parts[3])
                    dur = float(parts[4])
                    spk = parts[7]
                    ann[Segment(st, st + dur)] = spk
            return ann

        ref_ann = load_annotation(ref_path)
        hyp_ann = load_annotation(hyp_path)

        # 1. DER Collar 0.0s (Strict benchmark, skip_overlap=False)
        der_00_metric = DiarizationErrorRate(collar=0.0, skip_overlap=False)
        details_00 = der_00_metric(ref_ann, hyp_ann, detailed=True)
        total_time = float(details_00.get("total", 0.0) or 0.0)
        if total_time <= 0:
            raise ValueError(f"Reference annotation in {ref_path} has 0.0 total duration.")

        der_00 = float(details_00.get("diarization error rate", 0.0))
        miss_00 = float(details_00.get("missed detection", 0.0)) / total_time
        fa_00 = float(details_00.get("false alarm", 0.0)) / total_time
        conf_00 = float(details_00.get("confusion", 0.0)) / total_time

        # 2. DER Collar 0.25s (NIST forgiving benchmark, skip_overlap=False)
        der_25_metric = DiarizationErrorRate(collar=0.25, skip_overlap=False)
        der_25 = float(der_25_metric(ref_ann, hyp_ann))

        # 3. Jaccard Error Rate
        jer_metric = JaccardErrorRate()
        jer = float(jer_metric(ref_ann, hyp_ann))

        # 4. Speaker count error
        k_error_dict = {}
        if k_hat is not None and k_true is not None:
            k_error_dict = {
                "k_hat": int(k_hat),
                "k_true": int(k_true),
                "k_error": int(k_hat - k_true),
                "k_abs_error": int(abs(k_hat - k_true)),
            }

        # 5. Production cpWER calculation if aligned words and transcript are provided
        cpwer_metrics: Dict[str, Any] = {}
        if aligned_words_json is not None or reference_transcript_json is not None:
            if aligned_words_json is None:
                raise ValueError("aligned_words_json must be provided if reference_transcript_json is specified.")
            if reference_transcript_json is None:
                raise ValueError("reference_transcript_json must be provided if aligned_words_json is specified.")

            cpwer_metrics = evaluate_cpwer(
                hypothesis_rttm=hyp_path,
                aligned_words_json=aligned_words_json,
                reference_transcript_json=reference_transcript_json,
            )

        return {
            "der_collar0": round(der_00, 4),
            "miss_collar0": round(miss_00, 4),
            "fa_collar0": round(fa_00, 4),
            "conf_collar0": round(conf_00, 4),
            "der_collar025": round(der_25, 4),
            "jer": round(jer, 4),
            "total_speech_time_sec": round(total_time, 2),
            **k_error_dict,
            **cpwer_metrics,
        }
