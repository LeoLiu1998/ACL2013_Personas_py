from typing import *
from containers import *
import numpy as np
from sklearn.linear_model import LogisticRegression
from jproperties import Properties
import random
import sys
import traceback
import PrintUtil
import DataReader
import HyperparameterOptimization
from SparseMulticlassLogisticRegression import SparseMulticlassLogisticRegression


class IterInfo(object):
    def __init__(self):
        self.runningLL: float = 0.0
        self.numClassChanges: int = 0
        self.numFrameChanges: int = 0
        self.frameRunningLL: float = 0.0
        self.runningTotalLL: float = 0.0

    def __str__(self) -> str:
        return f"runningTotalLL {self.runningTotalLL} numchanges {self.numClassChanges}"

    def __repr__(self) -> str:
        return self.__str__()


class PersonaModel:
    def __init__(self):
        self.data: List[Doc] or None = None
        self.alpha: float = 1.0

        # indirect linkers: persona x verbclass
        self.LAgentTotal: np.ndarray or None = None  #  List[int]
        self.LAgents: np.ndarray or None = None  # List[List[int]]
        self.LPatientTotal: np.ndarray or None = None  # List[int]
        self.LPatients: np.ndarray or None = None  # List[List[int]]
        self.LModifierTotal: np.ndarray or None = None  # List[int]
        self.LModifiers: np.ndarray or None = None  # List[List[int]]

        # corresponding hyperparameters
        self.nuA: float = 1
        self.nuP: float = 1
        self.nuM: float = 1

        self.A: int = 50
        self.personaRegression: bool or None = None

        self.L2: float or None = None

        self.totalEntities: int or None = None

        self.genreIdToString: Dict[str, str] or None = None
        self.numFeatures: int or None = None

        # number of personas
        self.K: int or None = None
        # max vocabulary size
        self.V: int or None = None

        # hyperparameter on phi
        self.gamma: float = 1.0

        # verbclass x vocab multinomial
        self.phis: np.ndarray or None = None

        # verbclass token total counts
        self.phiTotals: np.ndarray or None = None

        # regression
        self.weights: str or None = None

        # map from vocab ID (index of array) to String
        self.reverseVocab: List[str] or None = None

        # trained logreg model
        self.model: SparseMulticlassLogisticRegression.TrainedModel or None = None

        self.finalPhis: np.ndarray or None = None
        self.finalLAgents: np.ndarray or None = None
        self.finalLPatients: np.ndarray or None = None
        self.finalLMod: np.ndarray or None = None

        self.characterNames: Dict[str, str] or None = None
        self.characterFeatures: Dict[str, Set[int]] or None = None

        self.featureMeans: np.ndarray or None = None

        self.featureIds: Dict[str, int] = {}
        self.reverseFeatureIds: List[str] or None = None

        # map from doc ids (1577389) to movie titles ("The Remains of the Day")
        self.movieTitles: Dict[str, str] or None = None

        # genre metadata for all the movies
        self.movieGenres: Dict[str, Set[str]] or None = None

        # for logistic regression: set of predictors, response counts (for each of K classes)
        # totalEntities x A
        self.responses: np.ndarray or None = None

        self.reg: SparseMulticlassLogisticRegression or None = None

        self.random: random = random.random()

    def initialize(self) -> None:
        self.LAgentTotal = np.full(self.A, 0)
        self.LAgents = np.full((self.A, self.K), 0)

        self.LPatientTotal = np.full(self.A, 0)
        self.LPatients = np.full((self.A, self.K), 0)

        self.LModifierTotal = np.full(self.A, 0)
        self.LModifiers = np.full((self.A, self.K), 0)

        self.phis = np.full((self.A, self.V), 0)
        self.phiTotals = np.full(self.K, 0)

        self.finalPhis = np.full((self.K, self.V), 0)
        self.finalLAgents = np.full((self.A, self.K), 0)
        self.finalLPatients = np.full((self.A, self.K), 0)
        self.finalLMod = np.full((self.A, self.K), 0)

        self.responses = np.full((self.totalEntities, self.A), 0)

        self.reg = SparseMulticlassLogisticRegression(self.L2)
        self.model = self.reg.getDefault(self.A, self.numFeatures)

    def saveSamples(self) -> None:

        for doc in self.data:
            for e in doc.entities.values():
                e: Entity
                # it's possible that some entities have no Events
                if e.getNumEvents() > 0:
                    self.responses[e.canonicalIndex][e.currentType] += 1
                    e.saveSample(e.currentType)

    def saveFinalSamples(self) -> None:
        for doc in self.data:
            for e in doc.entities.values():
                e: Entity
                a: int = e.currentType
                for arg in e.agentArgs:
                    arg.saveFinalSample()
                    self.finalLAgents[a][arg.currentSample] += 1
                    self.finalPhis[arg.currentSample][arg.tuple.canonicalVerb] += 1

                for arg in e.patientArgs:
                    arg.saveFinalSample()
                    self.finalLPatients[a][arg.currentSample] += 1
                    self.finalPhis[arg.currentSample][arg.tuple.canonicalVerb] += 1

                for arg in e.modifieeArgs:
                    arg.saveFinalSample()
                    self.finalLMod[a][arg.currentSample] += 1
                    self.finalPhis[arg.currentSample][arg.tuple.canonicalVerb] += 1

                if e.getNumEvents() > 0:
                    e.saveFinalSample(e.currentType)

    def regress(self) -> None:
        """
        Run multiclass logistic regression using all samples in responses. Clear
        response counts at the end.
        """
        def hash_ndarray(x: np.ndarray):
            return hash(tuple(x))

        regressionResponses: np.ndarray = np.zeros((self.totalEntities, self.A))
        regressionPredictors: np.ndarray= np.zeros((self.totalEntities, self.A))

        vectors: Dict[str, np.ndarray] = {}
        indices: Dict[int, int] = {}  # this is different from original java code because ndarray is not hashable
        maxInt: int = 0

        for doc in self.data:
            for e in doc.entities.values():
                if e.getNumEvents() > 0:
                    index: int = e.canonicalIndex
                    predictor: np.ndarray = np.zeros(self.numFeatures)

                    sampleCount: int = np.sum(self.responses[index]).item()

                    if sampleCount > 0:
                        if doc.genres is not None:
                            genreIDs = [self.featureIds[genre] for genre in doc.genres]

                            predictor[genreIDs] = 1.0
                            predictor[list(e.getCharacterFeatures())] = 1.0

                        # Group everything by unique combinations of features
                        key = str(predictor)
                        if key in vectors:
                            source_index: int = indices[hash_ndarray(vectors[key])]
                            keeper: np.ndarray = regressionResponses[source_index]

                            keeper += np.sum(self.responses[index], axis=1)
                            regressionResponses[source_index] = keeper
                        else:
                            regressionPredictors[maxInt] = predictor
                            vectors[str(predictor)] = predictor
                            indices[hash_ndarray(predictor)] = maxInt
                            regressionResponses[maxInt] = self.responses[index]
                            maxInt += 1

        # convert arraylist of responses to double array
        regressionResponses: np.ndarray = regressionResponses[:maxInt]
        regressionPredictors: np.ndarray = regressionPredictors[:maxInt]

        self.featureMeans = np.zeros(self.numFeatures)

        regressionPredictors[regressionPredictors != 0] = 1

        self.featureMeans = np.sum(regressionPredictors, axis=1) / self.totalEntities

        self.model = self.reg.regress(regressionPredictors, regressionResponses, self.numFeatures)
        self.model.printWeightsToFile(self.weights, self.reverseFeatureIds, self.genreIdToString)

        self.responses = np.zeros((self.totalEntities, self.A))

    def setDocumentPriors(self) -> None:
        """
        Once the model is trained, set the document priors for entities in the doc and
        the doc itself (which don't change between LogReg() runs).
        """
        for doc in self.data:
            doc: Doc
            for e in doc.entities.values():
                e: Entity
                if e.getNumEvents() > 0:

                    docPredictor: np.ndarray = np.zeros(self.numFeatures)
                    characterPredictor: np.ndarray = np.zeros(self.numFeatures)

                    if doc.genres is not None:
                        genreIDs = [self.featureIds[genre] for genre in doc.genres]
                        docPredictor[genreIDs] = 1.0
                        characterPredictor[genreIDs] = 1.0

                        characterPredictor[list(e.getCharacterFeatures())] = 1.0

                    docDistribution: np.ndarray = self.model.predict(docPredictor)
                    characterDistribution: np.ndarray = self.model.predict(characterPredictor)

                    e.prior = characterDistribution
                    doc.prior = docDistribution

    def generateConditionalPosterior(self) -> None:
        """
        Conditioning on an entity's persona mode, generate posteriors over the
	    typed topics given the observed words. (These are not used anywhere, but
	    could be interesting to analyze how a character's actions deviates from
	    the most likely persona).
        """
        samples: int = 100

        for doc in self.data:
            doc: Doc
            for e in doc.entities.values():
                e: Entity
                posterior: np.ndarray = e.posterior
                mode: int = np.argmax(posterior).item()

                conditionalAgentSamples: np.ndarray = np.full(self.K, 0)
                for arg in e.agentArgs:
                    prob: np.ndarray = np.full(self.K, 1.0)

                    for j in range(self.K):
                        prob[j] *= self.unaryLFactor(arg, j, mode)
                        prob[j] *= self.unaryEmissionFactor(arg, j)

                    prob = prob / np.sum(prob)
                    new_zs = np.random.choice(self.K, samples, p=prob)
                    conditionalAgentSamples[new_zs] += 1

                # normalize
                conditionalAgentNorm = np.sum(conditionalAgentSamples)
                if conditionalAgentNorm > 0: # TODO bug? 1?
                    conditionalAgentSamples = conditionalAgentSamples / conditionalAgentNorm
                e.conditionalAgentPosterior = conditionalAgentSamples

                conditionalPatientSamples: np.ndarray = np.full(self.K, 0.0)
                for arg in e.patientArgs:
                    prob: np.ndarray = np.full(self.K, 1.0)

                    for j in range(self.K):
                        prob[j] *= self.unaryLFactor(arg, j, mode)
                        prob[j] *= self.unaryEmissionFactor(arg, j)

                    prob = prob / np.sum(prob)
                    new_zs = np.random.choice(self.K, samples, p=prob)
                    conditionalPatientSamples[new_zs] += 1

                conditionalPatientNorm = np.sum(conditionalPatientSamples)
                if conditionalPatientNorm > 0:  # TODO bug? 1?
                    conditionalPatientSamples = conditionalPatientSamples / conditionalPatientNorm
                e.conditionalPatientPosterior = conditionalPatientSamples

                conditionalModSamples: np.ndarray = np.full(self.K, 0.0)
                for arg in e.modifieeArgs:
                    prob: np.ndarray = np.full(self.K, 1.0)

                    for j in range(self.K):
                        prob[j] *= self.unaryLFactor(arg, j, mode)
                        prob[j] *= self.unaryEmissionFactor(arg, j)

                    prob = prob / np.sum(prob)
                    new_zs = np.random.choice(self.K, samples, p=prob)
                    conditionalModSamples[new_zs] += 1

                conditionalModNorm = np.sum(conditionalModSamples)
                if conditionalModNorm > 0:  # TODO bug? 1?
                    conditionalModSamples = conditionalModSamples / conditionalModNorm
                e.conditionalModPosterior = conditionalModSamples

    def generatePosteriors(self) -> None:
        """
        Generate an entity's posterior distribution over personas from saved
        samples; and then wipe those saved samples for the next iteration.
        """
        for doc in self.data:
            for e in doc.entities.values():
                e.posterior = e.getSamplePosterior()
                e.posteriorSamples = np.full(self.A, 0.0)

    ############## z-level sampling inference stuff here  #######################
    #############################################################################

    def incrementClassInfo(self,
                           delta: int,
                           arg: EventArg,
                           cur_z: int,
                           currentType: int) -> None:

        self.incrementUnaries(delta,
                              cur_z,
                              arg.tuple.getCanonicalVerb(),
                              currentType,
                              arg.role)

    def incrementUnaries(self,
                         delta: int,
                         currentZ: int,
                         wordId: int,
                         currentType: int,
                         role: EventRole) -> None:
        self.phis[currentZ, wordId] += delta
        self.phiTotals[currentZ] += delta

        if role == EventRole.AGENT:
            self.LAgentTotal[currentType] += delta
            self.LAgents[currentType, currentZ] += delta

        if role == EventRole.PATIENT:
            self.LPatientTotal[currentType] += delta
            self.LPatients[currentType, currentZ] += delta

        if role == EventRole.MODIFIEE:
            self.LModifierTotal[currentType] += delta
            self.LModifierTotal[currentType, currentZ] += delta


    def unaryLFactor(self, arg: EventArg, k: int, type: int) -> float:
        # eq. (2) first term
        if arg.role == EventRole.AGENT:
            norm: float = self.LAgentTotal[type] + (self.K * self.nuA)
            return (self.LAgentTotal[type][k] + self.nuA) / norm

        if arg.role == EventRole.PATIENT:
            norm: float = self.LPatientTotal[type] + (self.K * self.nuP)
            return (self.LPatientTotal[type][k] + self.nuP) / norm

        if arg.role == EventRole.MODIFIEE:
            norm: float = self.LModifierTotal[type] + (self.K * self.nuM)
            return (self.LModifiers[type][k] + self.nuM) / norm

        return 1.0

    def unaryEmissionFactor(self, arg: EventArg, k: int) -> float:
        """
        p(w_unary | z=k)
        """
        verb: int = arg.tuple.getCanonicalVerb()
        norm: float = self.phiTotals[k] + (self.V * self.gamma)
        return (self.phis[k, verb] + self.gamma) / norm

    def LDASamplePersonas(self, first: bool) -> None:
        for doc in self.data:
            doc: Doc

            for entity in doc.entities.values():
                if not first:
                    doc.currentPersonaSamples[entity.currentType] -= 1.0

                ########## eq. (1) ##########
                regprobs = np.full(self.A, 1.0)

                for j in range(self.A):
                    # first term
                    regprobs[j] *= doc.currentPersonaSamples[j] + self.alpha

                    # second term
                    for e in entity.agentArgs:
                        regprobs[j] *= self.unaryLFactor(e, e.currentSample, j)
                    for e in entity.patientArgs:
                        regprobs[j] *= self.unaryLFactor(e, e.currentSample, j)
                    for e in entity.modifieeArgs:
                        regprobs[j] *= self.unaryLFactor(e, e.currentSample, j)

                # normalize the probability
                regprobs = regprobs / np.sum(regprobs)

                # draw a sample from multinomial
                new_z: int = np.random.choice(self.A, 1, p=regprobs).item()

                entity.lastType = entity.currentType
                entity.currentType = new_z

                doc.currentPersonaSamples[entity.currentType] += 1.0

    def LogRegSample(self) -> float:
        ll: float = 0.0
        for doc in self.data:
            for entity in doc.entities.values():
                characterPrior: np.ndarray = entity.prior

                regprobs: np.ndarray = np.full(self.A, 1.0)

                for j in range(self.A):
                    # first term
                    regprobs[j] *= characterPrior[j]

                    # second term
                    for e in entity.agentArgs:
                        regprobs[j] *= self.unaryLFactor(e, e.currentSample, j)
                    for e in entity.patientArgs:
                        regprobs[j] *= self.unaryLFactor(e, e.currentSample, j)
                    for e in entity.modifieeArgs:
                        regprobs[j] *= self.unaryLFactor(e, e.currentSample, j)

                # normalize the probability
                regprobs = regprobs / np.sum(regprobs)

                # draw a sample from multinomial
                new_z: int = np.random.choice(self.A, 1, p=regprobs).item()
                ll += np.log(characterPrior[new_z]).item()

                entity.lastType = entity.currentType
                entity.currentType = new_z
        return ll

    def sample(self, first: bool, doCompleteLL: bool) -> IterInfo:
        info: IterInfo = IterInfo()
        if personaRegression:
            info.runningTotalLL += self.LogRegSample()
        else:
            self.LDASamplePersonas(first)

        if doCompleteLL:
            info.runningTotalLL += HyperparameterOptimization.totalLL(model=self)

        for doc in self.data:
            for t in doc.eventTuples:
                t: EventTuple
                for arg in t.arguments.values():

                    old_z: int = arg.currentSample

                    if not first:
                        self.incrementClassInfo(-1, arg, old_z, arg.entity.lastType)

                    regprobs = np.full(self.K, 1.0)

                    for k in range(self.K):
                        regprobs[k] *= self.unaryLFactor(arg, k, arg.entity.currentType)

                        regprobs[k] *= self.unaryEmissionFactor(arg, k)

                    total: float = np.sum(regprobs).item()
                    if total <= 0:
                        print(f"0 sum {regprobs}")
                        sys.exit(1)

                    new_z: int = np.random.choice(self.K, 1, p=regprobs).item()

                    self.incrementClassInfo(1, arg, new_z, arg.entity.currentType)
                    arg.currentSample = new_z

                    assert regprobs[new_z] / total >= 0
                    info.numClassChanges += 1 if old_z != new_z else 0

        return info


if __name__ == "__main__":
    cli_args = sys.argv
    propertyFile = cli_args[0]

    gibbs: PersonaModel = PersonaModel()
    properties: Properties = Properties()

    try:
        properties.load(open(propertyFile), encoding=None)
        K: int = int(properties["K"].data)
        V: int = int(properties["V"].data)
        A: int = int(properties["A"].data)
        alpha: float = float(properties["alpha"].data)
        gamma: float = float(properties["gamma"].data)
        L2: float = float(properties["L2"].data)

        maxIterations: int = int(properties["maxIterations"].data) + 1

        dataFile: str = properties["data"].data
        movieMetadata: str = properties["movieMetadata"].data
        characterMetadata: str = properties["characterMetadata"].data
        characterPosteriorFile: str = properties["characterPosteriorFile"].data
        personaRegression: bool = bool(properties["personaRegression"].data)

        outPhiWeights: str = properties["outPhiWeights"].data
        featureMeans: str = properties["featureMeans"].data
        personaFile: str = properties["personaFile"].data
        characterConditionalPosteriorFile: str = properties["characterConditionalPosteriorFile"].data
        featureFile: str = properties["featureFile"].data
        finalLAgentsFile: str = properties["finalLAgentsFile"].data
        finalLPatientsFile: str = properties["finalLPatientsFile"].data
        finalLModFile: str = properties["finalLModFile"].data
        weights: str = properties["weights"].data

        gibbs.weights = weights
        gibbs.K = K
        gibbs.V = V
        gibbs.A = A
        gibbs.alpha = alpha
        gibbs.personaRegression = personaRegression
        gibbs.L2 = L2

        gibbs.gamma = gamma

        numWordsToPrint: int = 20

        DataReader.read(characterMetadata, movieMetadata, dataFile, gibbs)

        gibbs.initialize()

        # START TRAINING
        gibbs.sample(first=True, doCompleteLL=True)
        print("Iter 0: runningLL 0 numchanges 0")
        HyperparameterOptimization.resampleConcs(gibbs)

        burnin: int = maxIterations

        doCompleteLLEvery: int = 100
        regressEvery: int = 100  # test=100; real = 1000

        # take 100 samples
        for i in range(1000):
            if i % 10 == 0:
                print(f"{i} ")

            gibbs.sample(first=False, doCompleteLL=False)
            if i % 10 == 0:
                # save current a for all entities and z for all entityargs
                gibbs.saveFinalSamples()
        print()
        """
        Write character posteriors, conditional posteriors, and
        class/feature associations to file
        """

        PrintUtil.printFinalPosteriorsToFile(characterPosteriorFile,
                                             characterConditionalPosteriorFile,
                                             featureFile, gibbs)

        """Write personas to file"""
        PrintUtil.printFinalPersonas(personaFile, gibbs)

        print(PrintUtil.printMeanTop(gibbs.finalPhis,
                                     gibbs.reverseVocab,
                                     "final ratiorank", numWordsToPrint))
        print(PrintUtil.printSimpleTop(gibbs.finalPhis,
                                       gibbs.reverseVocab,
                                       "final freqrank", numWordsToPrint))

        names: List[str] = [str(i) for i in range(K)]

        """Write distributions to file"""
        PrintUtil.printWeights(gibbs.finalPhis,
                               gibbs.reverseVocab,
                               outPhiWeights)
        PrintUtil.printWeights(gibbs.finalLAgents,
                               names,
                               finalLAgentsFile)
        PrintUtil.printWeights(gibbs.finalLPatients,
                               names,
                               finalLPatientsFile)
        PrintUtil.printWeights(gibbs.finalLMod, names, finalLModFile)

    except FileNotFoundError:
        exc_info = sys.exc_info()
        traceback.print_exception(*exc_info)
    except IOError:
        exc_info = sys.exc_info()
        traceback.print_exception(*exc_info)