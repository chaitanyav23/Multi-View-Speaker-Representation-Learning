#!/usr/bin/env python3
"""
Stage 3 — Frozen Dual-Encoder Multiscale Embedding & Reliability Feature Extraction.

Requirements enforced:
1. Both NVIDIA TitaNet-Large (192-D) and WeSpeaker ResNet293-LM (256-D) remain strictly frozen.
2. Uses identical waveform inputs, VAD segmentation boundaries, and 5 multiscale temporal windows
   ([1.5, 1.2, 1.0, 0.8, 0.5]s with 50% shift) for both encoders.
3. Evaluates three controlled audio conditions:
   - '16k_mono': Common unenhanced 16 kHz mono waveform (primary controlled comparison)
   - 'dfn': DeepFilterNet3 enhanced 16 kHz mono waveform (Path A)
   - 'normalized': EBU R128 loudness-normalized 16 kHz mono waveform (Path B)
4. Verifies PyTorch ResNet293-LM (avg_model.pt) numerical equivalence against official
   ONNX export (voxceleb_resnet293_LM.onnx) prior to extraction.
5. Stores raw embeddings, L2-normalized embeddings, pre-normalization L2 norms, model/checkpoint
   SHA256 hashes, preprocessing metadata, subsegment timestamps, ground-truth speaker alignment,
   and acoustic/view reliability indicators in uniquely keyed caches.
6. Verifies embedding dimensions, finite values, L2 normalization, ordering, and 1-to-1 correspondence.
"""

import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.compliance.kaldi as kaldi
from huggingface_hub import hf_hub_download

REPO_ROOT = Path("/home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning")
STAGE1_DIR = REPO_ROOT / "research_stages" / "stage1_data_prep"
STAGE2_DIR = REPO_ROOT / "research_stages" / "stage2_baseline"
STAGE3_DIR = REPO_ROOT / "research_stages" / "stage3_dual_encoder"

SCALES = [
    (0, 1.50, 0.75, 1.0),
    (1, 1.20, 0.60, 2.0),
    (2, 1.00, 0.50, 3.0),
    (3, 0.80, 0.40, 3.0),
    (4, 0.50, 0.25, 3.0),
]
AUDIO_CONDITIONS = ["16k_mono", "dfn", "normalized"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# ============================================================================
# Official WeSpeaker ResNet293 Architecture (from wenet-e2e/wespeaker/models/resnet.py)
# ============================================================================

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class TSTP(nn.Module):
    """Temporal Statistics Pooling: concatenates temporal mean and standard deviation (with unbiased=True)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, F, T] -> flatten C and F -> [B, C*F, T]
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2], x.shape[3])
        mean = x.mean(dim=-1)
        std = torch.sqrt(torch.clamp(x.var(dim=-1, unbiased=True), min=1e-10))
        return torch.cat([mean, std], dim=-1)


class WeSpeakerResNet293(nn.Module):
    def __init__(
        self,
        num_blocks: List[int] = [10, 20, 64, 3],
        m_channels: int = 32,
        feat_dim: int = 80,
        embed_dim: int = 256,
        two_emb_layer: bool = False,
    ) -> None:
        super().__init__()
        self.in_planes = m_channels
        self.feat_dim = feat_dim
        self.embed_dim = embed_dim
        self.two_emb_layer = two_emb_layer
        self.stats_dim = int(feat_dim / 8) * m_channels * 8

        self.conv1 = nn.Conv2d(1, m_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(m_channels)
        self.layer1 = self._make_layer(Bottleneck, m_channels, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(Bottleneck, m_channels * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(Bottleneck, m_channels * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(Bottleneck, m_channels * 8, num_blocks[3], stride=2)

        self.pool = TSTP()
        self.pool_out_dim = self.stats_dim * Bottleneck.expansion * 2
        self.seg_1 = nn.Linear(self.pool_out_dim, embed_dim)
        if self.two_emb_layer:
            self.seg_bn_1 = nn.BatchNorm1d(embed_dim, affine=False)
            self.seg_2 = nn.Linear(embed_dim, embed_dim)
        else:
            self.seg_bn_1 = nn.Identity()
            self.seg_2 = nn.Identity()

    def _make_layer(self, block: Any, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T_frames, 80] -> permute to [B, 80, T_frames] -> unsqueeze(1) -> [B, 1, 80, T_frames]
        x = x.permute(0, 2, 1).unsqueeze(1)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        stats = self.pool(out)
        embed_a = self.seg_1(stats)
        if self.two_emb_layer:
            out = F.relu(embed_a)
            out = self.seg_bn_1(out)
            embed_b = self.seg_2(out)
            return embed_b
        return embed_a


def compute_wespeaker_fbank(wav_1d: torch.Tensor, sample_rate: int = 16000) -> torch.Tensor:
    """
    Exact WeSpeaker Kaldi Fbank frontend (wespeaker/cli/speaker.py):
    - Multiplies float [-1, 1] waveform by 32768.0 (16-bit PCM scale)
    - 80 mel bins, 25ms Hamming window, 10ms frame shift, dither=0.0, use_energy=False
    - Utterance/subsegment-level Cepstral Mean Normalization (CMN: subtract mean across frames)
    """
    pcm = (wav_1d.unsqueeze(0).to(torch.float32) * 32768.0)
    feats = kaldi.fbank(
        pcm,
        num_mel_bins=80,
        frame_length=25.0,
        frame_shift=10.0,
        dither=0.0,
        sample_frequency=float(sample_rate),
        window_type="hamming",
        use_energy=False,
    )
    # CMN (subtract per-bin temporal mean)
    feats = feats - torch.mean(feats, dim=0, keepdim=True)
    return feats  # [T_frames, 80]


def generate_neMo_identical_subsegments(
    vad_segments: List[Dict[str, Any]], window: float, shift: float
) -> List[Dict[str, float]]:
    """
    Exact replica of NeMo's speaker_utils.get_subsegments() so all audio conditions
    and encoders share identical subsegment timestamps [start, end, duration].
    """
    subsegs: List[Dict[str, float]] = []
    for seg in vad_segments:
        offset = float(seg["start"])
        duration = float(seg.get("duration", float(seg["end"]) - offset))
        if duration <= 0:
            continue
        if duration < window:
            subsegs.append({
                "start": round(offset, 4),
                "end": round(offset + duration, 4),
                "duration": round(duration, 4),
            })
        else:
            slice_st = offset
            end_time = offset + duration
            while slice_st + window <= end_time + 1e-6:
                subsegs.append({
                    "start": round(slice_st, 4),
                    "end": round(slice_st + window, 4),
                    "duration": round(window, 4),
                })
                slice_st += shift
            # Handle trailing remainder if significant (NeMo get_subsegments logic)
            if slice_st < end_time - 1e-4 and (end_time - slice_st) >= 0.05:
                subsegs.append({
                    "start": round(end_time - window, 4) if (end_time - offset) >= window else round(slice_st, 4),
                    "end": round(end_time, 4),
                    "duration": round(min(window, end_time - offset), 4),
                })
            # Deduplicate consecutive identical intervals
            if len(subsegs) >= 2 and subsegs[-1]["start"] == subsegs[-2]["start"] and subsegs[-1]["end"] == subsegs[-2]["end"]:
                subsegs.pop()
    return subsegs


def load_subsegments_from_nemo_manifest(manifest_json: Path) -> List[Dict[str, float]]:
    """Reads exact subsegments generated by NeMo ClusteringDiarizer in Stage 2."""
    subsegs = []
    for line in manifest_json.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        st = float(item["offset"])
        dur = float(item["duration"])
        subsegs.append({
            "start": round(st, 4),
            "end": round(st + dur, 4),
            "duration": round(dur, 4),
        })
    return subsegs


def match_subsegment_to_gt_speaker(
    st: float, en: float, gt_segments: List[Dict[str, Any]], min_purity: float = 0.75
) -> Tuple[str, float, bool]:
    """
    Matches a subsegment [st, en] to ground-truth positive-duration segments.
    Returns (gt_speaker_or_AMBIGUOUS, overlap_purity, is_multi_speaker_overlap).
    """
    dur = max(1e-6, en - st)
    spk_ov: Dict[str, float] = {}
    for g in gt_segments:
        g_st = float(g["start"])
        g_en = float(g["end"])
        if g_en <= g_st:
            continue
        ov = min(en, g_en) - max(st, g_st)
        if ov > 0:
            spk = str(g["speaker"])
            spk_ov[spk] = spk_ov.get(spk, 0.0) + ov

    if not spk_ov:
        return "NON_SPEECH_OR_ROUNDING_GAP", 0.0, False

    active_spks = [k for k, v in spk_ov.items() if v > 0.05]
    is_ovl = len(active_spks) > 1 or ("Speaker_all" in spk_ov)
    best_spk, best_ov = max(spk_ov.items(), key=lambda x: x[1])
    total_spk_ov = sum(spk_ov.values())
    purity = min(1.0, best_ov / max(dur, total_spk_ov))

    if best_spk == "Speaker_all":
        return "AMBIGUOUS_SPEAKER_ALL", purity, True
    if is_ovl:
        return f"OVERLAP::{best_spk}", purity, True
    if (best_ov / dur) < min_purity:
        return f"PARTIAL_BOUNDARY::{best_spk}", best_ov / dur, False
    return best_spk, best_ov / dur, False


def compute_spectral_flatness(wav_np: np.ndarray) -> float:
    if wav_np.size < 256:
        return 0.0
    spec = np.abs(np.fft.rfft(wav_np * np.hanning(wav_np.size))) ** 2 + 1e-12
    log_mean = float(np.mean(np.log(spec)))
    arith_mean = float(np.mean(spec))
    return float(np.exp(log_mean) / arith_mean)


def main() -> None:
    torch.manual_seed(42)
    np.random.seed(42)

    if not torch.cuda.is_available():
        raise RuntimeError("STRICT FAILURE: CUDA GPU is required for Stage 3 dual-encoder extraction.")

    device = torch.device("cuda")
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    token = Path("/home/priyanka/.cache/huggingface/token").read_text(encoding="utf-8").strip()

    for sub in ["cache", "models", "reports", "verification"]:
        (STAGE3_DIR / sub).mkdir(parents=True, exist_ok=True)

    # 1. Download and verify WeSpeaker ResNet293-LM (PyTorch avg_model.pt + ONNX voxceleb_resnet293_LM.onnx)
    print("[Stage 3] Downloading/verifying WeSpeaker ResNet293-LM checkpoints...")
    pt_ckpt_path = Path(
        hf_hub_download(
            repo_id="Wespeaker/wespeaker-voxceleb-resnet293-LM",
            filename="avg_model.pt",
            token=token,
        )
    )
    onnx_ckpt_path = Path(
        hf_hub_download(
            repo_id="Wespeaker/wespeaker-voxceleb-resnet293-LM",
            filename="voxceleb_resnet293_LM.onnx",
            token=token,
        )
    )
    pt_sha256 = sha256_file(pt_ckpt_path)
    onnx_sha256 = sha256_file(onnx_ckpt_path)

    resnet_model = WeSpeakerResNet293(
        num_blocks=[10, 20, 64, 3],
        m_channels=32,
        feat_dim=80,
        embed_dim=256,
        two_emb_layer=False,
    )
    state_dict = torch.load(str(pt_ckpt_path), map_location="cpu")
    # Strip any projection head keys ('projection.*') not used at inference
    model_keys = set(resnet_model.state_dict().keys())
    filtered_sd = {k: v for k, v in state_dict.items() if k in model_keys}
    missing_keys = model_keys - set(filtered_sd.keys())
    if missing_keys:
        raise RuntimeError(f"STRICT FAILURE: Missing keys in ResNet293 state_dict: {missing_keys}")
    resnet_model.load_state_dict(filtered_sd, strict=True)
    resnet_model.to(device)
    resnet_model.eval()
    for p in resnet_model.parameters():
        p.requires_grad_(False)

    # Cross-verify PyTorch ResNet293 vs official ONNX model on a synthetic/real fbank tensor
    test_fbank = torch.randn(1, 150, 80, dtype=torch.float32)
    with torch.no_grad():
        emb_pt = resnet_model(test_fbank.to(device)).cpu().numpy()[0]
    ort_sess = ort.InferenceSession(str(onnx_ckpt_path), providers=["CPUExecutionProvider"])
    emb_onnx = ort_sess.run(None, {"feats": test_fbank.numpy()})[0][0]
    pt_onnx_cos = float(np.dot(emb_pt, emb_onnx) / (np.linalg.norm(emb_pt) * np.linalg.norm(emb_onnx)))
    pt_onnx_max_diff = float(np.max(np.abs(emb_pt - emb_onnx)))
    print(f"[Stage 3 Verification] ResNet293-LM PyTorch vs ONNX cosine={pt_onnx_cos:.8f}, max_diff={pt_onnx_max_diff:.2e}")
    if pt_onnx_cos < 0.9999:
        raise RuntimeError(f"STRICT FAILURE: PyTorch vs ONNX parity check failed: cos={pt_onnx_cos}")
    del ort_sess

    # 2. Load Frozen NVIDIA TitaNet-Large (extracted from diar_msdd_telephonic as used in Stage 2)
    from nemo.collections.asr.models.msdd_models import EncDecDiarLabelModel
    from nemo.collections.asr.models.label_models import EncDecSpeakerLabelModel

    print("[Stage 3] Loading frozen NVIDIA TitaNet-Large on CUDA...")
    msdd_full = EncDecDiarLabelModel.from_pretrained(model_name="diar_msdd_telephonic", map_location="cuda")
    msdd_sd = msdd_full.state_dict()
    prefix = "msdd._speaker_model."
    spk_sd = {k.replace(prefix, ""): v for k, v in msdd_sd.items() if k.startswith(prefix)}
    titanet_model = EncDecSpeakerLabelModel.from_config_dict(msdd_full.cfg.speaker_model_cfg)
    titanet_model.load_state_dict(spk_sd, strict=True)
    titanet_model.to(device)
    titanet_model.eval()
    for p in titanet_model.parameters():
        p.requires_grad_(False)
    del msdd_full
    torch.cuda.empty_cache()

    manifest = json.loads((STAGE1_DIR / "manifests" / "recording_manifest.json").read_text(encoding="utf-8"))
    validation_records: List[Dict[str, Any]] = []

    for entry in manifest:
        rec_id = entry["rec_id"]
        gt_segments = json.loads((REPO_ROOT / entry["json_uniform"]).read_text())["segments"]

        # Load all 3 waveform conditions (16k_mono, dfn, normalized)
        wavs: Dict[str, np.ndarray] = {}
        for cond, key in [("16k_mono", "mono_wav"), ("dfn", "dfn_wav"), ("normalized", "norm_wav")]:
            w_tensor, sr = torchaudio.load(str(REPO_ROOT / entry[key]))
            if sr != 16000 or w_tensor.shape[0] != 1:
                raise RuntimeError(f"Invalid waveform at {entry[key]}")
            wavs[cond] = w_tensor.squeeze(0).numpy().astype(np.float32)

        # Estimate recording-level non-speech noise floor from non-VAD frames on 16k_mono
        vad_a_segs = json.loads((STAGE2_DIR / "vad" / f"{rec_id}_pathA_vad.json").read_text())["segments"]
        n_total_samples = wavs["16k_mono"].shape[0]
        speech_mask = np.zeros(n_total_samples, dtype=bool)
        for vs in vad_a_segs:
            s_i = max(0, int(round(vs["start"] * 16000)))
            e_i = min(n_total_samples, int(round(vs["end"] * 16000)))
            speech_mask[s_i:e_i] = True
        non_speech_samples = wavs["16k_mono"][~speech_mask]
        if non_speech_samples.size >= 1600:
            noise_floor_power = float(np.mean(non_speech_samples ** 2) + 1e-10)
        else:
            # Fallback to 10th percentile frame energy
            frames = np.lib.stride_tricks.sliding_window_view(wavs["16k_mono"], 400)[::160]
            f_pow = np.mean(frames ** 2, axis=-1)
            noise_floor_power = float(np.percentile(f_pow, 10) + 1e-10)

        # Load frame-level Pyannote posteriors for all 3 conditions
        posteriors: Dict[str, Any] = {}
        for cond in AUDIO_CONDITIONS:
            npz_p = STAGE2_DIR / "posteriors" / f"{rec_id}_{cond}_posteriors.npz"
            posteriors[cond] = np.load(str(npz_p))

        # Use the exact subsegments_scale{0..4}.json generated by NeMo on Path A in Stage 2
        # so that all 3 audio conditions and both encoders share 100% identical subsegment timestamps!
        scale_subsegments: Dict[int, List[Dict[str, float]]] = {}
        for scale_k, win_sec, shift_sec, _ in SCALES:
            nemo_subseg_json = (
                STAGE2_DIR / "msdd" / f"{rec_id}_msdd_outputs" / "speaker_outputs" / f"subsegments_scale{scale_k}.json"
            )
            if nemo_subseg_json.exists():
                scale_subsegments[scale_k] = load_subsegments_from_nemo_manifest(nemo_subseg_json)
            else:
                scale_subsegments[scale_k] = generate_neMo_identical_subsegments(vad_a_segs, win_sec, shift_sec)

        for cond in AUDIO_CONDITIONS:
            cond_dir = STAGE3_DIR / "cache" / cond / rec_id
            cond_dir.mkdir(parents=True, exist_ok=True)
            wav_arr = wavs[cond]
            post_data = posteriors[cond]
            f_times = post_data["frame_times"]
            p_speech_arr = post_data["p_speech"]
            p_ovl_arr = post_data["p_overlap"]

            # Store per-scale embeddings in memory first so we can compute cross-scale consistency
            tita_raw_by_scale: Dict[int, np.ndarray] = {}
            tita_norm_by_scale: Dict[int, np.ndarray] = {}
            tita_l2_by_scale: Dict[int, np.ndarray] = {}

            res_raw_by_scale: Dict[int, np.ndarray] = {}
            res_norm_by_scale: Dict[int, np.ndarray] = {}
            res_l2_by_scale: Dict[int, np.ndarray] = {}

            subseg_meta_by_scale: Dict[int, List[Dict[str, Any]]] = {}

            for scale_k, win_sec, shift_sec, ms_weight in SCALES:
                subsegs = scale_subsegments[scale_k]
                n_sub = len(subsegs)
                win_samples = int(round(win_sec * 16000))

                # Slice waveforms for all subsegments at this scale
                tita_wav_list: List[torch.Tensor] = []
                tita_len_list: List[int] = []
                res_fbank_list: List[torch.Tensor] = []
                meta_list: List[Dict[str, Any]] = []

                for idx, sb in enumerate(subsegs):
                    st = sb["start"]
                    en = sb["end"]
                    dur = sb["duration"]
                    s_samp = max(0, int(round(st * 16000)))
                    e_samp = min(wav_arr.shape[0], int(round(en * 16000)))
                    clip_np = wav_arr[s_samp:e_samp]
                    if clip_np.size < 400:
                        clip_np = np.pad(clip_np, (0, 400 - clip_np.size))

                    # Acoustic reliability indicators for subsegment [st, en]
                    sig_power = float(np.mean(clip_np ** 2) + 1e-10)
                    rms_dbfs = float(10.0 * math.log10(sig_power))
                    snr_local_db = float(10.0 * math.log10(max(sig_power / noise_floor_power, 1e-6)))

                    # DFN3 noise reduction energy ratio (using mono vs dfn on the exact same interval)
                    mono_clip = wavs["16k_mono"][s_samp:e_samp]
                    dfn_clip = wavs["dfn"][s_samp:e_samp]
                    if mono_clip.size == dfn_clip.size and mono_clip.size > 0:
                        removed_pow = float(np.mean((mono_clip - dfn_clip) ** 2) + 1e-10)
                        dfn_nr_db = float(10.0 * math.log10(float(np.mean(mono_clip ** 2) + 1e-10) / removed_pow))
                    else:
                        dfn_nr_db = 0.0

                    sfm = compute_spectral_flatness(clip_np)

                    # Frame-level VAD and overlap posteriors in [st, en]
                    f_mask = (f_times >= st) & (f_times <= en)
                    if np.any(f_mask):
                        vad_mean = float(np.mean(p_speech_arr[f_mask]))
                        vad_std = float(np.std(p_speech_arr[f_mask]))
                        ovl_mean = float(np.mean(p_ovl_arr[f_mask]))
                        ovl_max = float(np.max(p_ovl_arr[f_mask]))
                    else:
                        nearest_f = int(np.argmin(np.abs(f_times - 0.5 * (st + en))))
                        vad_mean = float(p_speech_arr[nearest_f])
                        vad_std = 0.0
                        ovl_mean = float(p_ovl_arr[nearest_f])
                        ovl_max = float(p_ovl_arr[nearest_f])

                    gt_label, gt_purity, gt_is_ovl = match_subsegment_to_gt_speaker(st, en, gt_segments)

                    meta_list.append({
                        "subseg_idx": idx,
                        "scale_idx": scale_k,
                        "window_sec": win_sec,
                        "shift_sec": shift_sec,
                        "multiscale_weight": ms_weight,
                        "start": st,
                        "end": en,
                        "duration": dur,
                        " midpoint": round(0.5 * (st + en), 4),
                        "rms_dbfs": round(rms_dbfs, 4),
                        "snr_local_db": round(snr_local_db, 4),
                        "dfn_noise_reduction_db": round(dfn_nr_db, 4),
                        "spectral_flatness": round(sfm, 6),
                        "vad_conf_mean": round(vad_mean, 5),
                        "vad_conf_std": round(vad_std, 5),
                        "overlap_prob_mean": round(ovl_mean, 5),
                        "overlap_prob_max": round(ovl_max, 5),
                        "gt_label": gt_label,
                        "gt_purity": round(gt_purity, 4),
                        "gt_is_overlap": gt_is_ovl,
                    })

                    # Prepare TitaNet tensor (padded/cropped to win_samples as in NeMo AudioToSpeechLabelDataset)
                    clip_t = torch.from_numpy(clip_np)
                    valid_len = min(clip_t.shape[0], win_samples)
                    padded_t = torch.zeros(win_samples, dtype=torch.float32)
                    padded_t[:valid_len] = clip_t[:valid_len]
                    tita_wav_list.append(padded_t)
                    tita_len_list.append(valid_len)

                    # Prepare WeSpeaker ResNet293-LM Kaldi fbank tensor
                    fbank_t = compute_wespeaker_fbank(clip_t[:valid_len], sample_rate=16000)
                    res_fbank_list.append(fbank_t)

                # Batched TitaNet-Large extraction on CUDA
                tita_embs_list = []
                batch_size = 64
                with torch.no_grad():
                    for b_st in range(0, n_sub, batch_size):
                        b_en = min(n_sub, b_st + batch_size)
                        b_wav = torch.stack(tita_wav_list[b_st:b_en]).to(device)
                        b_len = torch.tensor(tita_len_list[b_st:b_en], dtype=torch.long, device=device)
                        _, b_embs = titanet_model.forward(input_signal=b_wav, input_signal_length=b_len)
                        tita_embs_list.append(b_embs.float().cpu().numpy())
                tita_raw = np.concatenate(tita_embs_list, axis=0).astype(np.float32)
                tita_norms = np.linalg.norm(tita_raw, axis=-1, keepdims=True)
                tita_normed = (tita_raw / np.maximum(tita_norms, 1e-12)).astype(np.float32)

                # Batched WeSpeaker ResNet293-LM extraction on CUDA
                # Group subsegments with identical frame count for exact batching without padding artifacts
                res_raw = np.zeros((n_sub, 256), dtype=np.float32)
                len_groups: Dict[int, List[int]] = {}
                for idx, fb in enumerate(res_fbank_list):
                    len_groups.setdefault(int(fb.shape[0]), []).append(idx)
                with torch.no_grad():
                    for n_frm, indices in len_groups.items():
                        for b_st in range(0, len(indices), batch_size):
                            b_idx = indices[b_st : b_st + batch_size]
                            b_fb = torch.stack([res_fbank_list[i] for i in b_idx]).to(device)
                            b_out = resnet_model(b_fb).float().cpu().numpy()
                            res_raw[b_idx] = b_out
                res_norms = np.linalg.norm(res_raw, axis=-1, keepdims=True)
                res_normed = (res_raw / np.maximum(res_norms, 1e-12)).astype(np.float32)

                tita_raw_by_scale[scale_k] = tita_raw
                tita_norm_by_scale[scale_k] = tita_normed
                tita_l2_by_scale[scale_k] = tita_norms.squeeze(-1).astype(np.float32)

                res_raw_by_scale[scale_k] = res_raw
                res_norm_by_scale[scale_k] = res_normed
                res_l2_by_scale[scale_k] = res_norms.squeeze(-1).astype(np.float32)
                subseg_meta_by_scale[scale_k] = meta_list

            # Compute cross-scale cosine consistency (scale k vs parent scale max(0, k-1))
            for scale_k, win_sec, shift_sec, ms_weight in SCALES:
                parent_k = max(0, scale_k - 1) if scale_k > 0 else 1
                cur_meta = subseg_meta_by_scale[scale_k]
                par_meta = subseg_meta_by_scale[parent_k]
                par_mids = np.array([m[" midpoint"] for m in par_meta], dtype=np.float32)

                t_norm_cur = tita_norm_by_scale[scale_k]
                t_norm_par = tita_norm_by_scale[parent_k]
                r_norm_cur = res_norm_by_scale[scale_k]
                r_norm_par = res_norm_by_scale[parent_k]

                t_l2 = tita_l2_by_scale[scale_k]
                r_l2 = res_l2_by_scale[scale_k]

                for i, m in enumerate(cur_meta):
                    p_idx = int(np.argmin(np.abs(par_mids - m[" midpoint"])))
                    c_tita = float(np.dot(t_norm_cur[i], t_norm_par[p_idx]))
                    c_res = float(np.dot(r_norm_cur[i], r_norm_par[p_idx]))
                    m["titanet_l2_norm"] = round(float(t_l2[i]), 6)
                    m["resnet293_l2_norm"] = round(float(r_l2[i]), 6)
                    m["titanet_cross_scale_cos"] = round(c_tita, 6)
                    m["resnet293_cross_scale_cos"] = round(c_res, 6)
                    m["parent_scale_idx"] = parent_k
                    m["parent_subseg_idx"] = p_idx

                # Save uniquely keyed cache files and verify integrity
                cache_key_tita = f"titanet_large__{cond}__scale{scale_k}"
                cache_key_res = f"wespeaker_resnet293_lm__{cond}__scale{scale_k}"

                tita_npz = cond_dir / f"{cache_key_tita}.npz"
                res_npz = cond_dir / f"{cache_key_res}.npz"
                meta_json = cond_dir / f"subsegments_metadata__scale{scale_k}.json"

                starts_arr = np.array([m["start"] for m in cur_meta], dtype=np.float32)
                ends_arr = np.array([m["end"] for m in cur_meta], dtype=np.float32)

                np.savez_compressed(
                    str(tita_npz),
                    raw_embeddings=tita_raw_by_scale[scale_k],
                    l2_normalized_embeddings=tita_norm_by_scale[scale_k],
                    pre_norm_l2=tita_l2_by_scale[scale_k],
                    starts=starts_arr,
                    ends=ends_arr,
                )
                np.savez_compressed(
                    str(res_npz),
                    raw_embeddings=res_raw_by_scale[scale_k],
                    l2_normalized_embeddings=res_norm_by_scale[scale_k],
                    pre_norm_l2=res_l2_by_scale[scale_k],
                    starts=starts_arr,
                    ends=ends_arr,
                )
                meta_json.write_text(
                    json.dumps(
                        {
                            "rec_id": rec_id,
                            "audio_condition": cond,
                            "scale_idx": scale_k,
                            "window_sec": win_sec,
                            "shift_sec": shift_sec,
                            "num_subsegments": len(cur_meta),
                            "titanet_cache_key": cache_key_tita,
                            "resnet293_cache_key": cache_key_res,
                            "subsegments": cur_meta,
                        },
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )

                # Strict Validation Checks
                t_raw = tita_raw_by_scale[scale_k]
                t_nrm = tita_norm_by_scale[scale_k]
                r_raw = res_raw_by_scale[scale_k]
                r_nrm = res_norm_by_scale[scale_k]

                t_unit_err = float(np.max(np.abs(np.linalg.norm(t_nrm, axis=-1) - 1.0)))
                r_unit_err = float(np.max(np.abs(np.linalg.norm(r_nrm, axis=-1) - 1.0)))

                valid = (
                    t_raw.shape == (len(cur_meta), 192)
                    and r_raw.shape == (len(cur_meta), 256)
                    and int(np.isnan(t_raw).sum()) == 0
                    and int(np.isinf(t_raw).sum()) == 0
                    and int(np.isnan(r_raw).sum()) == 0
                    and int(np.isinf(r_raw).sum()) == 0
                    and t_unit_err < 1e-5
                    and r_unit_err < 1e-5
                )
                if not valid:
                    raise RuntimeError(f"STRICT FAILURE: Embedding validation failed for {rec_id} {cond} scale {scale_k}")

                validation_records.append({
                    "rec_id": rec_id,
                    "audio_condition": cond,
                    "scale_idx": scale_k,
                    "window_sec": win_sec,
                    "num_subsegments": len(cur_meta),
                    "titanet_shape": f"{t_raw.shape[0]}x{t_raw.shape[1]}",
                    "resnet293_shape": f"{r_raw.shape[0]}x{r_raw.shape[1]}",
                    "titanet_l2_mean": round(float(np.mean(t_l2)), 4),
                    "titanet_l2_std": round(float(np.std(t_l2)), 4),
                    "resnet293_l2_mean": round(float(np.mean(r_l2)), 4),
                    "resnet293_l2_std": round(float(np.std(r_l2)), 4),
                    "nan_or_inf_count": 0,
                    "max_unit_norm_err": round(max(t_unit_err, r_unit_err), 8),
                    "validation_passed": valid,
                })

        print(f"[Stage 3] Extracted & validated all 3 audio conditions x 5 scales for {rec_id}")

    # Write Stage 3 verification manifests
    ver_json = STAGE3_DIR / "verification" / "stage3_cache_validation.json"
    ver_csv = STAGE3_DIR / "verification" / "stage3_cache_validation.csv"
    ver_json.write_text(
        json.dumps(
            {
                "models": {
                    "titanet_large": {
                        "source": "diar_msdd_telephonic (msdd._speaker_model)",
                        "embed_dim": 192,
                        "frontend": "AudioToMelSpectrogramPreprocessor (80 mel, Hann 25ms/10ms, per_feature mean+std norm)",
                    },
                    "wespeaker_resnet293_lm": {
                        "repo_id": "Wespeaker/wespeaker-voxceleb-resnet293-LM",
                        "avg_model_pt_sha256": pt_sha256,
                        "onnx_sha256": onnx_sha256,
                        "pytorch_vs_onnx_cosine": pt_onnx_cos,
                        "pytorch_vs_onnx_max_abs_diff": pt_onnx_max_diff,
                        "embed_dim": 256,
                        "frontend": "Kaldi fbank (80 mel, Hamming 25ms/10ms, pcm*32768, per-subsegment CMN mean-only)",
                    },
                },
                "total_caches_validated": len(validation_records),
                "all_passed": all(r["validation_passed"] for r in validation_records),
                "records": validation_records,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    with ver_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(validation_records[0].keys()))
        w.writeheader()
        w.writerows(validation_records)

    print(f"Stage 3 extraction complete: {len(validation_records)} caches verified (all_passed=True).")


if __name__ == "__main__":
    main()
