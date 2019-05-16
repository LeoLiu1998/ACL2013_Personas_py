from typing import *
import numpy as np
from Entity import Entity
from EventTuple import EventTuple


class Doc(object):
    def __init__(self, A: int, personaRegression: bool):
        self.id: str or None = None
        self.title: str or None = None
        self.genres: Set[str] or None = None
        self.entities: Dict[str, Entity] or None = None
        self.eventTuples: List[EventTuple] = []
        self.prior: np.ndarray = np.full(A, 1.0/A)
        self.currentPersonaSamples: List[int]
        if not personaRegression:
            self.currentPersonaSamples = np.full(A, 0)
