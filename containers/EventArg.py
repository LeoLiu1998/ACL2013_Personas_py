from typing import *
import numpy as np
from containers import *


class EventArg(object):
    def __init__(self, role: EventRole, entity: Entity):
        self.role: EventRole = role
        self.entity: Entity = entity

        self.currentSample: int = -1
        self.finalSamples: Dict[int, int] = {}

        self.tuple: EventTuple or None = None  # the enclosing tuple, included only for convenience for display routines

    def getVerb(self) -> str:
        return self.tuple.getVerbString()

    def saveFinalSample(self) -> None:
        count = 0
        if self.currentSample in self.finalSamples:
            # if curSample is present, increment, else add a new entry
            count = self.finalSamples[self.currentSample]
        count += 1
        self.finalSamples[self.currentSample] = count

    def getFinalSamples(self, K: int):
        samples = np.full(K, 0)
        # for k, v in self.finalSamples.items():
            # samples[k] = v
        samples[list(self.finalSamples.keys())] = list(self.finalSamples.values())
        return samples