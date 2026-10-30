# Phase 14 Error Analysis

A critical bug was identified in the `step6_fusion.py` output logic. While `bwc_msdd.rttm` correctly identifies two speakers, the `bwc_final.txt` generator incorrectly maps both to `Speaker_1`. This is due to a dictionary mapping error where the first encountered speaker index persists across all turns.