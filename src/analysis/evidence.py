#!/usr/bin/env python3
"""
src/analysis/evidence.py

Định nghĩa cấu trúc dữ liệu chuẩn cho Acoustic Chain of Evidence (ACoE).
Bao gồm các nhóm chứng cứ:
  - Classifier Evidence: P(bonafide), P(spoof), prediction
  - Spectral Evidence: Centroid, Rolloff, Bandwidth, Flatness, Flux (mean, std)
  - Temporal Evidence: RMS, ZCR, Silence ratio, Voiced ratio
  - Cepstral Evidence: MFCC, Delta, Delta-Delta (mean, std vectors)
  - Prosodic Evidence: F0 mean, std, range, voiced ratio (hoặc status: unavailable)
  - Challenges Result: Kết quả biến đổi âm học có kiểm soát
  - Stability Evidence: Điểm nhất quán âm học (Acoustic Consistency Score)
  - Metadata: Tần số lấy mẫu, thời lượng, trạng thái xử lý
"""

from dataclasses import asdict, dataclass, field
import json
from typing import Any, Dict, List, Optional


@dataclass
class ClassifierEvidence:
    p_bonafide: float
    p_spoof: float
    prediction: str  # "bonafide" hoặc "spoof"


@dataclass
class SpectralEvidence:
    centroid_mean: float
    centroid_std: float
    rolloff_mean: float
    rolloff_std: float
    bandwidth_mean: float
    bandwidth_std: float
    flatness_mean: float
    flatness_std: float
    flux_mean: float
    flux_std: float


@dataclass
class TemporalEvidence:
    rms_mean: float
    rms_std: float
    rms_variation: float
    zcr_mean: float
    zcr_std: float
    silence_ratio: float
    voiced_ratio_est: float


@dataclass
class CepstralEvidence:
    mfcc_mean: List[float]
    mfcc_std: List[float]
    delta_mean: List[float]
    delta_std: List[float]
    delta2_mean: List[float]
    delta2_std: List[float]


@dataclass
class ProsodicEvidence:
    status: str  # "available" hoặc "unavailable"
    f0_mean: Optional[float] = None
    f0_std: Optional[float] = None
    f0_range: Optional[float] = None
    voiced_ratio: Optional[float] = None
    pitch_variation: Optional[float] = None


@dataclass
class ChallengeResult:
    name: str
    parameters: Dict[str, Any]
    p_spoof: float
    delta_p_spoof: float
    evidence_delta: float


@dataclass
class StabilityEvidence:
    acoustic_consistency_score: float  # Điểm [0, 1]
    challenge_count: int
    mean_abs_delta_p_spoof: float
    mean_evidence_delta: float
    per_challenge_stability: Dict[str, float] = field(default_factory=dict)


@dataclass
class MetadataEvidence:
    sample_rate: int
    duration_seconds: float
    processing_status: str  # "success", "partial", "failed"
    error_message: Optional[str] = None


@dataclass
class AcousticChainOfEvidence:
    """
    Bản ghi cấu trúc hoàn chỉnh của chuỗi chứng cứ âm học (Acoustic Chain of Evidence - ACoE).
    """
    file_id: str
    classifier: ClassifierEvidence
    spectral: SpectralEvidence
    temporal: TemporalEvidence
    cepstral: CepstralEvidence
    prosodic: ProsodicEvidence
    challenges: List[ChallengeResult]
    stability: StabilityEvidence
    metadata: MetadataEvidence

    def to_dict(self) -> Dict[str, Any]:
        """Chuyển đổi sang dict tương thích JSON theo đúng schema quy định."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Xuất chuỗi JSON có định dạng."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
