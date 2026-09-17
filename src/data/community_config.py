"""Community configuration and domain parameters for Anacity Facility Usage Simulation."""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class FacilityConfig:
    name: str
    display_name: str
    capacity_per_slot: int
    open_hour: int
    close_hour: int
    base_popularity_weight: float
    # Log-normal parameters (mean and sigma of log(hours))
    lead_time_log_mean: float
    lead_time_log_sigma: float
    min_lead_time_hours: float = 0.5
    max_lead_time_hours: float = 336.0  # 14 days


COMMUNITY_FACILITIES: Dict[str, FacilityConfig] = {
    "Gym": FacilityConfig(
        name="Gym",
        display_name="Gym",
        capacity_per_slot=40,
        open_hour=6,
        close_hour=22,
        base_popularity_weight=0.38,
        lead_time_log_mean=2.5,  # median ~12 hours
        lead_time_log_sigma=0.6,
        min_lead_time_hours=1.0,
        max_lead_time_hours=48.0,
    ),
    "Pool": FacilityConfig(
        name="Pool",
        display_name="Pool",
        capacity_per_slot=25,
        open_hour=6,
        close_hour=21,
        base_popularity_weight=0.24,
        lead_time_log_mean=2.8,  # median ~16.4 hours
        lead_time_log_sigma=0.7,
        min_lead_time_hours=2.0,
        max_lead_time_hours=72.0,
    ),
    "Badminton": FacilityConfig(
        name="Badminton",
        display_name="Badminton",
        capacity_per_slot=4,  # 2 courts x 2 slots
        open_hour=6,
        close_hour=22,
        base_popularity_weight=0.18,
        lead_time_log_mean=3.1,  # median ~22.2 hours
        lead_time_log_sigma=0.5,
        min_lead_time_hours=4.0,
        max_lead_time_hours=96.0,
    ),
    "Tennis": FacilityConfig(
        name="Tennis",
        display_name="Tennis",
        capacity_per_slot=4,  # 1 court
        open_hour=6,
        close_hour=21,
        base_popularity_weight=0.11,
        lead_time_log_mean=3.4,  # median ~30.0 hours
        lead_time_log_sigma=0.6,
        min_lead_time_hours=6.0,
        max_lead_time_hours=120.0,
    ),
    "Clubhouse": FacilityConfig(
        name="Clubhouse",
        display_name="Clubhouse",
        capacity_per_slot=30,
        open_hour=10,
        close_hour=23,
        base_popularity_weight=0.06,
        lead_time_log_mean=3.9,  # median ~49.4 hours (~2 days)
        lead_time_log_sigma=0.8,
        min_lead_time_hours=12.0,
        max_lead_time_hours=168.0,
    ),
    "Hall": FacilityConfig(
        name="Hall",
        display_name="Multipurpose Hall",
        capacity_per_slot=150,  # 1 event booking at a time
        open_hour=9,
        close_hour=23,
        base_popularity_weight=0.03,
        lead_time_log_mean=5.0,  # median ~148 hours (~6 days)
        lead_time_log_sigma=0.6,
        min_lead_time_hours=48.0,
        max_lead_time_hours=336.0,
    ),
}


@dataclass
class ResidentPersona:
    archetype: str
    weight: float
    preferred_facilities: Dict[str, float]
    preferred_days: List[int]  # 0 = Mon, ..., 6 = Sun
    preferred_hours: List[int]  # Valid hours of the day
    booking_cadence_days_mean: float
    noise_tolerance: float


RESIDENT_PERSONAS: List[ResidentPersona] = [
    ResidentPersona(
        archetype="Dawn Athletes",
        weight=0.25,
        preferred_facilities={"Gym": 0.70, "Tennis": 0.20, "Pool": 0.10},
        preferred_days=[0, 1, 2, 3, 4],  # Weekdays
        preferred_hours=[6, 7, 8],
        booking_cadence_days_mean=2.0,
        noise_tolerance=0.05,
    ),
    ResidentPersona(
        archetype="Evening Sports",
        weight=0.25,
        preferred_facilities={"Badminton": 0.55, "Gym": 0.35, "Tennis": 0.10},
        preferred_days=[0, 1, 2, 3, 4, 5],
        preferred_hours=[18, 19, 20, 21],
        booking_cadence_days_mean=2.5,
        noise_tolerance=0.06,
    ),
    ResidentPersona(
        archetype="Weekend Leisure",
        weight=0.20,
        preferred_facilities={"Pool": 0.60, "Clubhouse": 0.30, "Gym": 0.10},
        preferred_days=[5, 6],  # Sat, Sun
        preferred_hours=[10, 11, 14, 15, 16, 17, 18],
        booking_cadence_days_mean=6.0,
        noise_tolerance=0.08,
    ),
    ResidentPersona(
        archetype="Sporadic Casual",
        weight=0.20,
        preferred_facilities={"Gym": 0.30, "Pool": 0.30, "Badminton": 0.20, "Tennis": 0.15, "Clubhouse": 0.05},
        preferred_days=[0, 1, 2, 3, 4, 5, 6],
        preferred_hours=[7, 8, 12, 17, 18, 19],
        booking_cadence_days_mean=9.0,
        noise_tolerance=0.20,
    ),
    ResidentPersona(
        archetype="Community Organizers",
        weight=0.10,
        preferred_facilities={"Hall": 0.50, "Clubhouse": 0.35, "Pool": 0.15},
        preferred_days=[4, 5, 6],  # Fri, Sat, Sun
        preferred_hours=[11, 12, 15, 16, 18, 19],
        booking_cadence_days_mean=14.0,
        noise_tolerance=0.10,
    ),
]
