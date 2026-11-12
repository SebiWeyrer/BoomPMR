#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#
# Details:  Standalone replicate of the convergence analysis ("Ca") of
#           'dynamicsLab.py' (cell "do convergence analysis to get good amount
#           of ANCF and beam elements"). The analysis is repeated for several
#           Newton settings: (1) the standard solver settings used in the work
#           and (2) decreased Newton tolerances. All runs are plotted in one
#           figure using the same visualization settings as in the work.
# Author:   Sebastian Weyrer
# Date:     2026-11-12
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import sys
import pickle
import numpy as np
import time as tm
import exudyn as exu
import matplotlib.pyplot as plt
from pathlib import Path
srcDir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(srcDir)) # to import beams and utilities from src
from exudyn.utilities import *
from beams import EulerBernulliBeam, BeamTemplate
from utilities import *
print('EXUDYN says hello with version', exu.config.Version() + '.')

#++++++++++++++++++++++++++++++++++++++++++++++++++
# settings of this script
onlyLoadResults = False # if True, results are loaded from resultsFileName (skip simulations)
doBeamNR = True         # if False, only the ANCF convergence analysis is performed
saveFigures = False
resultsFileName = Path(__file__).resolve().parent / 'resultsConvergenceAnalysisNewtonTolerances.pkl'
figPath = Path(__file__).resolve().parent

#++++++++++++++++++++++++++++++++++++++++++++++++++
# Newton settings to be investigated (None ... keep standard settings of GetStandardSimulationAndSetVisualizationSettings)
# EXUDYN defaults: newton.relativeTolerance = 1e-8, newton.absoluteTolerance = 1e-10
newtonSettingsList = [{'name': 'standard tol.', 'relativeTolerance': None, 'absoluteTolerance': None},
                      {'name': 'decreased tol.', 'relativeTolerance': 1e-10, 'absoluteTolerance': 1e-12}]

#++++++++++++++++++++++++++++++++++++++++++++++++++
# default setup of beam reference parameters (same as in dynamicsLab.py)
length = 5          # [m] length of the beams
nElementsBeam = 50  # from convergence analysis
beamTemplateRubber = BeamTemplate(physicsLength=length,
                                  physicsAxialStiffness=2e7, physicsBendingStiffness=5e4,
                                  physicsMassPerLength=78, nElements=nElementsBeam,
                                  doNotInterpolateForSlidingBC=False)

#++++++++++++++++++++++++++++++++++++++++++++++++++
# time parameters of simulation (same as in dynamicsLab.py)
tEnd = 15           # [s] simulation duration
tExperiment = tEnd  # the experiment duration is equal the simulation duration
h = 1e-3            # [s] step size

#++++++++++++++++++++++++++++++++++++++++++++++++++
# some general settings
textWidth = 5.15
SetMatplotlibSettings()
standardColors, labelList, rawDataLabelList, markerList, timeStr, slidingStr = GetStandardValues()

#++++++++++++++++++++++++++++++++++++++++++++++++++
# EVOLVED test case (same as in dynamicsLab.py)
startPosition = length/4        # [m] start position of slider
stopPosition = length*(3/4)     # [m] stop position of slider
startTime = tExperiment/4       # [s] start of test
stopTime = tExperiment*(3/4)    # [s] end of test
A = 50                          # [N] max. amplitude of load functions
def LoadTipFunctionEvolved(mbs, t, load):
    fStart = 1
    fEnd = 10
    tChirp = tExperiment / 2
    if t <= tChirp:
        phase = 2 * np.pi * (fStart * t + (fEnd - fStart) * (t**2 / (2 * tChirp)))
    else:
        tLocal = t - tChirp
        phaseAtSwitch = 2 * np.pi * (fStart * tChirp + (fEnd - fStart) * (tChirp / 2))
        phase = phaseAtSwitch + 2 * np.pi * (fEnd * tLocal - (fEnd - fStart) * (tLocal**2 / (2 * tChirp)))
    return A * np.sin(phase)
def SlidingJointFunctionEvolved(t, startPosition=startPosition, stopPosition=stopPosition,
                               startTime=startTime, stopTime=stopTime):
    if t <= startTime:
        return startPosition
    if t >= stopTime:
        return startPosition
    else:
        duration = (stopTime-startTime) / 2
        if t > startTime and t <= (startTime+duration):
            return startPosition + ((stopPosition-startPosition)/(duration)) * (t-startTime)
        else:
            return stopPosition - ((stopPosition-startPosition)/(duration)) * (t-(startTime+duration))

#++++++++++++++++++++++++++++++++++++++++++++++++++
# helper function to get the simulation settings incl. the requested Newton tolerances
def GetSimulationSettings(newtonSettings):
    simulationSettings = GetStandardSimulationAndSetVisualizationSettings(tEnd, h, exportFrames=False)
    if newtonSettings['relativeTolerance'] is not None:
        simulationSettings.timeIntegration.newton.relativeTolerance = newtonSettings['relativeTolerance']
    if newtonSettings['absoluteTolerance'] is not None:
        simulationSettings.timeIntegration.newton.absoluteTolerance = newtonSettings['absoluteTolerance']
    return simulationSettings

#%% do convergence analysis for all Newton settings
nElementsCaANCFList = np.linspace(2, 44, 22, dtype=int)
nElementsCaNRList = np.linspace(2, 60, 30, dtype=int)
if not onlyLoadResults:
    resultsList = []
    for newtonSettings in newtonSettingsList:
        PrintHeadlineToConsole(text='Convergence Analysis: ' + newtonSettings['name'])
        print('relativeTolerance:', newtonSettings['relativeTolerance'], '| absoluteTolerance:', newtonSettings['absoluteTolerance'])
        # initialize beam (to inherit some data to ANCF beam)
        beamCaNR = EulerBernulliBeam(beamTemplate=beamTemplateRubber, surpressWarningsUponInitialization=True)

        #++++++++++++++++++++++++++++++++++++++++++++++++++
        # ANCF elements
        timeStartAnalyses = tm.time()
        nAnalysesCompleted = 0
        nAnalyses = len(nElementsCaANCFList)
        tipDisplacementCaANCFList = []
        for nElementsCaANCF in nElementsCaANCFList:
            SC = exu.SystemContainer()
            mbs = SC.AddSystem()
            nGround = mbs.AddNode(NodePointGround())
            # this is needed to constrain ANCF beam on the 'left' side
            mCoordGround = mbs.AddMarker(MarkerNodeCoordinate(nodeNumber=nGround, coordinate=0,
                                                              visualization=VMarkerNodeCoordinates(show=False)))
            sensorOutputVariableTypeCaANCFList = [exu.OutputVariableType.Displacement]
            sDictCaANCF = AddANCFBeamWithSlidingJointToMBSAndAddSensors(mbs, SlidingJointFunctionEvolved, LoadTipFunctionEvolved, length=length, rhoA=beamCaNR.rhoA, EI=beamCaNR.EI, EA=beamCaNR.EA,
                                                                        nElementsANCF=nElementsCaANCF, mCoordGround=mCoordGround, sensorOutputVariableTypeANCFList=sensorOutputVariableTypeCaANCFList,
                                                                        beamNodeReferencePositionList=np.array([length])) # only one sensor at tip
            mbs.Assemble()
            mbs.SolveDynamic(simulationSettings=GetSimulationSettings(newtonSettings))
            outputDictCaANCF = GetOutputDictANCFFromSDictANCF(mbs, sDictANCF=sDictCaANCF, sensorOutputVariableTypeANCFList=sensorOutputVariableTypeCaANCFList)
            tipDisplacementCaANCFList += [outputDictCaANCF['OutputVariableType.Displacement'][:, -2]]
            nAnalysesCompleted += 1
            ProgressTracker(timeStartAnalyses, nAnalyses, nAnalysesCompleted, showETA=True)

        #++++++++++++++++++++++++++++++++++++++++++++++++++
        # beam elements
        tipDisplacementCaNRList = []
        if doBeamNR:
            timeStartAnalyses = tm.time()
            nAnalysesCompleted = 0
            nAnalyses = len(nElementsCaNRList)
            for nElementsCaNR in nElementsCaNRList:
                SC = exu.SystemContainer()
                mbs = SC.AddSystem()
                nGround = mbs.AddNode(NodePointGround())
                mCoordsGround = mbs.AddMarker(MarkerNodeCoordinates(nodeNumber=nGround, visualization=VMarkerNodeCoordinates(show=False)))
                beamCaNR = EulerBernulliBeam(beamTemplate=beamTemplateRubber,
                                             nElements=nElementsCaNR, # OVERWRITE!
                                             slidingJointFunction=SlidingJointFunctionEvolved,
                                             loadTipFunction=LoadTipFunctionEvolved)
                sCoordinatesCaNR, sCoordinates_tCaNR = beamCaNR.AddBeamToMBSAndAddSensors(mbs, mCoordsGround=mCoordsGround, show=False)
                mbs.Assemble()
                mbs.SolveDynamic(simulationSettings=GetSimulationSettings(newtonSettings))
                _, time, qCaNR, q_tCaNR = beamCaNR.GetTimeAndCoordinates(mbs, sCoordinatesCaNR, sCoordinates_tCaNR)
                tipDisplacementCaNRList += [qCaNR[:, -2]]
                nAnalysesCompleted += 1
                ProgressTracker(timeStartAnalyses, nAnalyses, nAnalysesCompleted, showETA=True)
        resultsList += [{'newtonSettings': newtonSettings,
                         'nElementsCaANCFList': nElementsCaANCFList,
                         'tipDisplacementCaANCFList': tipDisplacementCaANCFList,
                         'nElementsCaNRList': nElementsCaNRList,
                         'tipDisplacementCaNRList': tipDisplacementCaNRList}]
        with open(resultsFileName, 'wb') as f: # save after each setting (long runs)
            pickle.dump(resultsList, f)
else:
    with open(resultsFileName, 'rb') as f:
        resultsList = pickle.load(f)

#%% compute errors w.r.t. finest discretization and plot all runs in one figure
def GetErrorList(tipDisplacementList):
    errorList = [AbsError(tipDisplacement, tipDisplacementList[-1]) for tipDisplacement in tipDisplacementList[:-1]]
    return errorList + [0]
lineStyleList = ['-', '--', ':', '-.']
figCa, axCa = plt.subplots(1, 1, figsize=[textWidth, 2.2])
for i, results in enumerate(resultsList):
    lineStyle = lineStyleList[i % len(lineStyleList)]
    suffix = ' (' + results['newtonSettings']['name'] + ')'
    errorCaANCFList = GetErrorList(results['tipDisplacementCaANCFList'])
    nDOFsANCFList = (results['nElementsCaANCFList'] + 1) * 4 # 4 DOFs per node
    axCa.loglog(nDOFsANCFList[:-1], errorCaANCFList[:-1], lineStyle + 'o', label=labelList[0] + suffix,
                color=standardColors[0], fillstyle='full' if i == 0 else 'none')
    if len(results['tipDisplacementCaNRList']) > 0:
        errorCaNRList = GetErrorList(results['tipDisplacementCaNRList'])
        nDOFsNRList = (results['nElementsCaNRList'] + 1) * 3 # 3 DOFs per node
        axCa.loglog(nDOFsNRList[:-1], errorCaNRList[:-1], lineStyle + '*', label=labelList[1] + suffix,
                    color=standardColors[1], fillstyle='full' if i == 0 else 'none')
axCa.grid()
axCa.legend()
axCa.set_xlabel('number of DOFs')
axCa.set_ylabel('abs. error transv.\ntip displacement in m')
SaveFig(figCa, figPath, 'E_convergenceAnalysisNewtonTolerances', closeFig=False, saveFig=saveFigures)
plt.show()
