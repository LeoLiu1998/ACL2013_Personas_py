from typing import *
from containers import *


class EventTuple(object):
    def __init__(self, tupleID: str, verbString: str):
        self.tupleID: str = tupleID
        self.canonicalVerb: int = -1
        self.verbString: str = verbString
        self.arguments: Dict[EventRole, EventArg]= {}
        self.currentFrame: int = -1

    def numberize(self, verbVocab: Dict[str, int]) -> None:
        assert self.verbString in verbVocab
        self.canonicalVerb = verbVocab[self.verbString]

    def getArg(self, role: EventRole) -> Entity or None:
        if role in self.arguments:
            return self.arguments[role].entity
        return None

    def setCanonicalVerb(self, canonicalVerb: int) -> None:
        self.canonicalVerb = canonicalVerb

    def getCanonicalVerb(self) -> int:
        return self.canonicalVerb

    def getTupleID(self) -> str:
        return self.tupleID

    def setTupleID(self, tupleID: str) -> None:
        if self.tupleID is not None:
            assert self.tupleID == tupleID, "bug in loading code"
        self.tupleID = tupleID

    def setArg(self, arg: EventArg) -> None:
        self.arguments[arg.role] = arg

    def numArgs(self) -> int:
        return len(self.arguments)

    def setVerbString(self, verbString: str):
        self.verbString = verbString

    def getVerbString(self) -> str:
        return self.verbString

    def __str__(self) -> str:
        s = ""
        s += f"{self.verbString}({self.tupleID})\t"
        for r in EventRole:
            if r in self.arguments:
                s += f"\t{r}={self.getArg(r)}"
        s += "\t]"
        return s

    def __repr__(self) -> str:
        return self.__str__()