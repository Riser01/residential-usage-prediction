"""Synthetic historical facility booking dataset generator for Anacity platform.

Author: Prajwal Rao
Incorporates all 8 real-world dimensions specified in Anacity problem requirements:
1. Resident preferences (latent archetypes)
2. Facility popularity (Pareto distribution)
3. Time patterns (diurnal/weekly cycles)
4. Booking lead times (log-normal per facility/resident)
5. Sparsity (active vs dormant residents)
6. Imbalance (skewed facility & resident volumes)
7. Noise (exploratory & atypical bookings)
8. Changing behaviour (concept drift & move-ins)
"""

import math
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from src.data.community_config import (
    COMMUNITY_FACILITIES,
    FacilityConfig,
    RESIDENT_PERSONAS,
    ResidentPersona,
)


class CommunityDatasetGenerator:
    """Generates synthetic historical facility bookings adhering to community constraints."""

    def __init__(
        self,
        num_residents: int = 350,
        start_date: str = "2026-01-01",
        num_days: int = 180,  # 6 months (Jan 1 to Jun 30, 2026)
        seed: int = 42,
    ):
        self.num_residents = num_residents
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
        self.end_date = self.start_date + timedelta(days=num_days)
        self.num_days = num_days
        self.seed = seed

        # Set reproducible random seeds
        random.seed(seed)
        np.random.seed(seed)

        self.residents = [f"R-{100 + i}" for i in range(num_residents)]
        self._assign_resident_profiles()

    def _assign_resident_profiles(self) -> None:
        """Assigns latent archetypes, engagement tiers, and drift status to residents."""
        self.resident_profiles: Dict[str, dict] = {}
        persona_weights = [p.weight for p in RESIDENT_PERSONAS]

        for resident_id in self.residents:
            persona = random.choices(RESIDENT_PERSONAS, weights=persona_weights, k=1)[0]

            # Engagement Tier (Sparsity & Imbalance)
            tier_rand = random.random()
            if tier_rand < 0.20:
                tier = "heavy"
                cadence_multiplier = 0.45
            elif tier_rand < 0.50:
                tier = "medium"
                cadence_multiplier = 1.0
            else:
                tier = "sporadic"
                cadence_multiplier = 2.4

            lead_time_scale = np.random.lognormal(0.0, 0.25)

            # Move-in date: 90% exist at start; 10% move in after Day 90 (Cold-Start simulation)
            join_day = random.randint(90, 150) if random.random() < 0.10 else 0

            # Concept Drift: 15% of residents undergo behavioral change around Day 105
            has_drift = random.random() < 0.15
            drift_day = random.randint(95, 115) if has_drift else 9999

            self.resident_profiles[resident_id] = {
                "persona": persona,
                "tier": tier,
                "cadence_multiplier": cadence_multiplier,
                "lead_time_scale": lead_time_scale,
                "join_day": join_day,
                "has_drift": has_drift,
                "drift_day": drift_day,
            }

    def generate(self) -> pd.DataFrame:
        """Generates all booking records across the simulation horizon strictly matching PDF §3.1 schema."""
        records: List[dict] = []
        slot_occupancy: Dict[Tuple[str, datetime], int] = {}
        resident_schedule: Set[Tuple[str, datetime]] = set()

        for resident_id, profile in self.resident_profiles.items():
            persona: ResidentPersona = profile["persona"]
            current_day = profile["join_day"]
            cadence_mean = persona.booking_cadence_days_mean * profile["cadence_multiplier"]

            while current_day < self.num_days:
                step = max(1, int(np.random.exponential(scale=cadence_mean) + 0.5))
                current_day += step
                if current_day >= self.num_days:
                    break

                is_drifted = profile["has_drift"] and (current_day >= profile["drift_day"])
                facility_name = self._sample_facility(persona, is_drifted, current_day)
                facility_cfg = COMMUNITY_FACILITIES[facility_name]

                usage_date, usage_hour = self._sample_usage_slot(
                    persona, facility_cfg, current_day, is_drifted
                )
                usage_datetime = usage_date.replace(hour=usage_hour, minute=0, second=0)

                # Guard against double booking for resident
                if (resident_id, usage_datetime) in resident_schedule:
                    continue

                # Guard against facility capacity overflow
                current_occupancy = slot_occupancy.get((facility_name, usage_datetime), 0)
                if current_occupancy >= facility_cfg.capacity_per_slot:
                    continue

                # Sample lead time
                lead_time_hours = self._sample_lead_time(facility_cfg, profile["lead_time_scale"])
                booking_datetime = usage_datetime - timedelta(hours=lead_time_hours)

                if booking_datetime < self.start_date or booking_datetime >= usage_datetime:
                    continue

                # Snap booking minute to natural intervals (e.g. 18:15)
                booking_minute = random.choice([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55])
                booking_datetime = booking_datetime.replace(minute=booking_minute, second=0)

                if booking_datetime >= usage_datetime:
                    booking_datetime = usage_datetime - timedelta(minutes=random.randint(30, 180))

                # Commit booking
                slot_occupancy[(facility_name, usage_datetime)] = current_occupancy + 1
                resident_schedule.add((resident_id, usage_datetime))

                # Strictly the 4 core fields defined in Anacity PDF §3.1
                records.append(
                    {
                        "resident_id": resident_id,
                        "facility_id": facility_name,
                        "booking_timestamp": booking_datetime.strftime("%Y-%m-%d %H:%M:%S"),
                        "usage_timestamp": usage_datetime.strftime("%Y-%m-%d %H:%M:%S"),
                    }
                )

        df = pd.DataFrame(records)
        df["booking_dt"] = pd.to_datetime(df["booking_timestamp"])
        df = df.sort_values("booking_dt").reset_index(drop=True)
        df = df.drop(columns=["booking_dt"])

        return df

    def _sample_facility(self, persona: ResidentPersona, is_drifted: bool, current_day: int) -> str:
        """Samples facility incorporating preferences, noise, and drift."""
        if random.random() < persona.noise_tolerance:
            all_facs = list(COMMUNITY_FACILITIES.keys())
            weights = [COMMUNITY_FACILITIES[f].base_popularity_weight for f in all_facs]
            return random.choices(all_facs, weights=weights, k=1)[0]

        pref_map = dict(persona.preferred_facilities)

        if is_drifted:
            if "Gym" in pref_map:
                pref_map["Pool"] = pref_map.get("Pool", 0.0) + 0.40
                pref_map["Gym"] = max(0.05, pref_map["Gym"] - 0.40)
            elif "Weekend Leisure" in persona.archetype:
                pref_map["Badminton"] = pref_map.get("Badminton", 0.0) + 0.30

        if current_day > 105:
            pref_map["Pool"] = pref_map.get("Pool", 0.0) + 0.15

        facs = list(pref_map.keys())
        w = list(pref_map.values())
        return random.choices(facs, weights=w, k=1)[0]

    def _sample_usage_slot(
        self,
        persona: ResidentPersona,
        facility_cfg: FacilityConfig,
        day_offset: int,
        is_drifted: bool,
    ) -> Tuple[datetime, int]:
        """Samples target usage date and hour within facility operating hours."""
        base_date = self.start_date + timedelta(days=day_offset)

        if random.random() < 0.85:
            target_dow = random.choice(persona.preferred_days)
            diff = (target_dow - base_date.weekday()) % 7
            usage_date = base_date + timedelta(days=diff)
        else:
            usage_date = base_date

        valid_operating_hours = list(range(facility_cfg.open_hour, facility_cfg.close_hour + 1))
        matching_hours = [h for h in persona.preferred_hours if h in valid_operating_hours]

        if matching_hours and random.random() < 0.85:
            usage_hour = random.choice(matching_hours)
        else:
            usage_hour = random.choice(valid_operating_hours)

        if is_drifted and usage_hour < 10 and 19 in valid_operating_hours:
            if random.random() < 0.70:
                usage_hour = 19

        return usage_date, usage_hour

    def _sample_lead_time(self, facility_cfg: FacilityConfig, resident_scale: float) -> float:
        """Samples lead time using facility log-normal distribution."""
        raw_lead = np.random.lognormal(
            facility_cfg.lead_time_log_mean, facility_cfg.lead_time_log_sigma
        )
        lead = raw_lead * resident_scale
        lead = max(facility_cfg.min_lead_time_hours, min(facility_cfg.max_lead_time_hours, lead))
        return lead


def generate_and_save_dataset(
    output_path: str = "data/facility_bookings.csv",
    num_residents: int = 350,
    seed: int = 42,
) -> pd.DataFrame:
    """Convenience function to generate and persist dataset."""
    generator = CommunityDatasetGenerator(num_residents=num_residents, seed=seed)
    df = generator.generate()
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} bookings for {num_residents} residents -> {output_path}")
    return df


if __name__ == "__main__":
    generate_and_save_dataset()
