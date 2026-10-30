# Phase 20 Fusion Evaluation Refactor

## Literature followed

- MeetEval paper: von Neumann et al., "MeetEval: A Toolkit for Computation of Word Error Rates for Meeting Transcription Systems", CHiME 2023 / arXiv:2307.11394.
- MeetEval implementation methodology: cpWER uses speaker-wise concatenation, empty streams for speaker-count mismatch, Levenshtein costs, and Hungarian assignment.
- ESPnet and CHiME-7 DASR practice: diarization-attributed ASR is evaluated with permutation-invariant speaker matching and speaker-attributed transcript scoring.

No MeetEval code is imported or called at runtime. The Hungarian solver and Levenshtein edit counter are implemented locally in `scripts/speaker_mapping.py`.

## Mathematical definitions

- `cpwer`: concatenate words for every reference and hypothesis speaker; build pairwise Levenshtein edit-distance costs; solve the minimum one-to-one speaker assignment; normalize total selected edits by total reference words.
- `sawer`: apply the optimal hypothesis-to-reference speaker mapping; concatenate mapped hypothesis words per reference speaker; aggregate per-speaker substitutions, insertions, and deletions; normalize by total reference words.
- `speaker_accuracy`: preserve the existing Phase 20 definition. Timed hypothesis words are paired with the reference speaker at that time, a one-to-one speaker mapping maximizes correct pairs, and accuracy is `correct_attributions / words_evaluated`.
- `word_attribution_error`: defined only as `1 - speaker_accuracy`.
- `timeline_accuracy`: create 10 ms active-speaker frame sets for reference and mapped hypothesis timelines; report the fraction of frames with identical sets.

## Assumptions

- Text normalization lowercases, removes punctuation, collapses whitespace, and tokenizes on spaces, matching the project’s prior WER normalization intent.
- Empty reference transcripts produce WER `0.0`; missing hypothesis words against non-empty reference words produce deletion errors.
- Timeline overlap is represented as active-speaker sets, so overlapping speakers are not overwritten.
- Word-level speaker accuracy remains undefined (`None`) when no words can be evaluated.

## Compatibility

- Public API remains `fusion_metrics(gt_json, attributed_json, duration)` through `scripts/common.py`.
- New local modules are inside `phase20_standalone/scripts/`:
  - `speaker_mapping.py`
  - `cpwer.py`
  - `sawer.py`
  - `timeline_metrics.py`
  - `attribution_metrics.py`
  - `fusion_metrics.py`
- Fusion JSON now returns exactly:
  - `cpwer`
  - `sawer`
  - `speaker_accuracy`
  - `word_attribution_error`
  - `timeline_accuracy`
  - `words_evaluated`
  - `correct_attributions`
  - `speaker_mapping`
- Legacy fusion fields were removed from generated JSON and Markdown reports.

## Validation results

- Synthetic tests passed for 2, 3, and 4 speakers; speaker swaps; missing speaker; extra speaker; empty transcript; and overlapping speakers.
- BWC validation used only `ground_truth/bwc.txt` and `raw_audio/bwc.wav` as external validation assets. The temporary parsed validation JSON was created under `/tmp` and deleted.
- BWC validation metrics from the allowed text reference and standalone attributed output:
  - `cpwer`: 0.546
  - `sawer`: 0.546
  - `speaker_accuracy`: 0.8982630272952854
  - `word_attribution_error`: 0.1017369727047146
  - `timeline_accuracy`: 0.6939124487004104
  - `words_evaluated`: 403
  - `correct_attributions`: 362
- Standalone project artifact metrics regenerated from `phase20_standalone/ground_truth_json/bwc.json` and `phase20_standalone/fusion/attributed_words.json`:
  - `cpwer`: 0.5330578512396694
  - `sawer`: 0.5330578512396694
  - `speaker_accuracy`: 0.9027431421446384
  - `word_attribution_error`: 0.09725685785536164
  - `timeline_accuracy`: 0.70531235750114
  - `words_evaluated`: 401
  - `correct_attributions`: 362

## Final verification

- No imports from directories outside `phase20_standalone`.
- No runtime dependency on MeetEval.
- No SciPy dependency for assignment; Hungarian assignment is local.
- Fusion API is unchanged for callers.
- Removed legacy fusion fields from active JSON and Markdown outputs.
- Outputs are deterministic through sorted speaker keys and deterministic edit-distance tie breaking.
