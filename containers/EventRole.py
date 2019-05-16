from enum import Enum


class EventRole(Enum):
    AGENT, PATIENT, MODIFIEE= "Agent", "Patient", "Modifiee"

    def __str__(self) -> str:
        if self == EventRole.AGENT:
            return "Agent"
        if self == EventRole.PATIENT:
            return "Patient"
        if self == EventRole.MODIFIEE:
            return "Modifiee"
        assert False

    def __repr__(self):
        return self.__str__()

    def other(self):
        assert self == EventRole.AGENT or self == EventRole.PATIENT
        return EventRole.PATIENT if self == EventRole.AGENT else EventRole.AGENT
