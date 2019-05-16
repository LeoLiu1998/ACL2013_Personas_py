from typing import *
import numpy as np
from containers import *


class Entity(object):
    def __init__(self, id: str, A: int):
        self.id: str = id # E1 =
        self.name: str or None = None # Batman = None
        self.fullname: str or None = None  # Batman, who ....
        self.agentArgs: List[EventArg] = []
        self.patientArgs: List[EventArg] = []
        self.modifieeArgs: List[EventArg] = []

        self.canonicalIndex: int or None = None

        # current assignment of persona types (for EM setting)
        self.currentType: int = np.random.randint(0, A)
        self.lastType: int = self.currentType


        self.A: int = A

        self.prior: np.ndarray = np.full(A, 1.0/A)
        self.posterior: np.ndarray or None = None

        self.posteriorSamples: np.ndarray = np.full(A, 0.0)
        self.conditionalAgentPosterior: np.ndarray or None = None
        self.conditionalPatientPosterior: np.ndarray or None = None
        self.conditionalModPosterior: np.ndarray or None = None

        self.characterFeatures: Set[int] or None = None

        self.finalSamples: np.ndarray = np.full(A, 0.0)
        self.finalSampleSum: int = 0

    def saveSample(self, i: int) -> None:
        self.posteriorSamples[i] += 1

    def saveFinalSample(self, i: int) -> None:
        self.finalSamples[i] += 1
        self.finalSampleSum += 1

    def getCharacterFeatures(self) -> Set[int]:
        if self.characterFeatures is None:
            return set()
        return self.characterFeatures

    def getSamplePosterior(self) -> np.ndarray:
        total = np.sum(self.posteriorSamples)
        return self.posteriorSamples/total

    def getNumEvents(self) -> int:
        return len(self.agentArgs) + len(self.patientArgs) + len(self.modifieeArgs)

    def getEventString(self) -> str:
        buffer = "A: "
        for e in self.agentArgs:
            buffer += e.getVerb() + " "

        buffer += "P: "
        for e in self.agentArgs:
            buffer += e.getVerb() + " "
        return buffer

    def getMax(self) -> int:
        if self.finalSamples > 0:
            return int(np.argmax(self.finalSamples))
        return int(np.argmax(self.posterior))

    def getId(self) -> str:
        return self.id

    def getFullname(self) -> str:
        return self.fullname

    def getName(self) -> str:
        return self.name

    def setName(self, name: str) -> None:
        self.name = name

    def setFullname(self, fullname: str) -> None:
        self.fullname = fullname

    def setId(self, id: str) -> None:
        self.id = id

    def __str__(self) -> str:
        return f"{self.id}\t{self.name}\t{self.fullname}"

    def __repr__(self) -> str:
        return self.__str__()
