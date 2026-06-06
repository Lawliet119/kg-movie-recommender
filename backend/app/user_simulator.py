"""
Offline user simulator for KGenSam-style evaluation.

The simulator uses MovieLens-derived interaction profiles as ground truth:
- accepts an attribute when it appears in the user's positive profile
- rejects an attribute when it appears in the user's negative profile
- accepts a recommendation when the item is in the user's positive items
"""
from __future__ import annotations

from .interaction_data import UserInteractionProfile


class UserSimulator:
    def __init__(self, profile: UserInteractionProfile):
        self.profile = profile

    def answer_attribute(self, attr_type: str, attr_value: str) -> bool:
        key = f"{attr_type}:{attr_value}"
        if key in self.profile.positive_attributes:
            return True
        if key in self.profile.negative_attributes:
            return False

        # Unknown attributes are treated as negative feedback in offline eval.
        return False

    def accepts_recommendation(self, movie_id: str) -> bool:
        return movie_id in self.profile.positive_items
