import json
import pandas as pd
import numpy as np

def assign_speakers_to_words(aligned_res, msdd_segments, tolerance=0.5):
    """
    aligned_res: Output from whisperx.align. Expected to have 'segments' -> 'words'.
    msdd_segments: List of dicts [{'start': float, 'end': float, 'speaker': str}]
    """
    diagnostics = {
        "multiple_candidates": 0,
        "no_overlap": 0,
        "speaker_transitions": 0,
        "total_words": 0
    }
    
    word_metadata = []
    
    # Flatten words
    for seg in aligned_res["segments"]:
        for word_info in seg.get("words", []):
            if "start" not in word_info or "end" not in word_info:
                # Sometimes words lack timestamps if alignment failed. 
                # We skip them or fallback.
                continue
                
            diagnostics["total_words"] += 1
            w_start = word_info["start"]
            w_end = word_info["end"]
            w_mid = (w_start + w_end) / 2.0
            word_text = word_info["word"]
            
            # Find overlaps
            overlaps = []
            for msdd in msdd_segments:
                if msdd["start"] <= w_end and msdd["end"] >= w_start:
                    overlap_start = max(w_start, msdd["start"])
                    overlap_end = min(w_end, msdd["end"])
                    overlap_dur = overlap_end - overlap_start
                    if overlap_dur > 0:
                        overlaps.append({
                            "speaker": msdd["speaker"],
                            "overlap_dur": overlap_dur,
                            "msdd_start": msdd["start"],
                            "msdd_end": msdd["end"]
                        })
            
            assigned_spk = None
            overlap_duration = 0.0
            confidence = "high"
            
            if len(overlaps) == 1:
                # Case A
                assigned_spk = overlaps[0]["speaker"]
                overlap_duration = overlaps[0]["overlap_dur"]
            elif len(overlaps) > 1:
                # Case B
                diagnostics["multiple_candidates"] += 1
                diagnostics["speaker_transitions"] += 1 # Transition within word
                overlaps.sort(key=lambda x: x["overlap_dur"], reverse=True)
                assigned_spk = overlaps[0]["speaker"]
                overlap_duration = overlaps[0]["overlap_dur"]
                confidence = "medium"
            else:
                # Case C: No overlap
                diagnostics["no_overlap"] += 1
                # Find nearest
                nearest_spk = None
                min_dist = float('inf')
                for msdd in msdd_segments:
                    dist = min(abs(msdd["start"] - w_end), abs(msdd["end"] - w_start))
                    if dist < min_dist:
                        min_dist = dist
                        nearest_spk = msdd["speaker"]
                
                if min_dist <= tolerance and nearest_spk:
                    assigned_spk = nearest_spk
                    confidence = "low"
                else:
                    assigned_spk = "UNK"
                    confidence = "none"

            word_info["speaker"] = assigned_spk
            word_metadata.append({
                "word": word_text,
                "start": w_start,
                "end": w_end,
                "midpoint": w_mid,
                "assigned_speaker": assigned_spk,
                "overlap_duration": overlap_duration,
                "attribution_confidence": confidence
            })
            
    # Now assign speaker to the segment based on majority word speaker
    for seg in aligned_res["segments"]:
        speakers = []
        for w in seg.get("words", []):
            if "speaker" in w and w["speaker"] != "UNK":
                speakers.append(w["speaker"])
        
        if speakers:
            # Majority vote
            from collections import Counter
            seg["speaker"] = Counter(speakers).most_common(1)[0][0]
        else:
            seg["speaker"] = "UNK"
            
    return aligned_res, word_metadata, diagnostics
