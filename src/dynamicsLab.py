#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#
# Details:  
# Author:   Sebastian Weyrer
# Date:     2026-05-11
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import sys
import pickle
import socket
import platform
import numpy as np
import time as tm
import exudyn as exu
import matplotlib.pyplot as plt
from exudyn.utilities import *
from beams import EulerBernulliBeam, BeamTemplate
from utilities import *
from pathlib import Path
from datetime import datetime
print('EXUDYN says hello with version', exu.config.Version() + '.')

#++++++++++++++++++++++++++++++++++++++++++++++++++
# decide whether only to load results or produce them
onlyLoadResults = False # if False, expect long execution time due to expensive tests

#++++++++++++++++++++++++++++++++++++++++++++++++++
# setup of 'test' beams with which the experiments are made (and ANCF reference settings)
length = 5          # [m] length of the beams
nElementsANCF = 20  # from convergence analysis (see later)
nElementsBeam = 50  # from convergence analysis (see later)
beamTemplateRubber = BeamTemplate(physicsLength=length,
                                  physicsAxialStiffness=2e7, physicsBendingStiffness=5e4,
                                  physicsMassPerLength=78, nElements=nElementsBeam,
                                  doNotInterpolateForSlidingBC=False)
beamIllustration = EulerBernulliBeam(beamTemplate=beamTemplateRubber) # use this beam only for illustration purposes!

#++++++++++++++++++++++++++++++++++++++++++++++++++
# define time parameters of simulation
tEnd = 15           # [s] simulation duration
tExperiment = tEnd  # the experiment duration is equal the simulation duration
h = 1e-3            # [s] step size

#++++++++++++++++++++++++++++++++++++++++++++++++++
# some general settings
textWidth = 5.15
SetMatplotlibSettings()
standardColors, labelList, rawDataLabelList, markerList, timeStr, slidingStr = GetStandardValues()
parentDir = Path.cwd().parent
figPath = parentDir / 'LaTeX' / 'figures'

#++++++++++++++++++++++++++++++++++++++++++++++++++
# settings to write results file
resultsFileName = 'results.pkl'
if not onlyLoadResults:
    PrintHeadlineToConsole(text='Creating empty results file to save results')
    data = {
        'metaInfo': {
            'created': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'hostname': socket.gethostname(),
            'platform': platform.platform(),
            'pythonVersion': sys.version,
            'exudynVersion': exu.config.Version(),
        },
        'results': {}
    }
    with open(resultsFileName, 'wb') as f:
        pickle.dump(data, f)
else:
    with open(resultsFileName, 'rb') as f:
        data = pickle.load(f)
    PrintHeadlineToConsole(text='Loading Existing Results')
    for info in data['metaInfo']:
        print(info + ': ' + data['metaInfo'][info])
    
#%% plot interpolation of y displacement within beam element
PrintHeadlineToConsole(text='Plot Example Interpolation for Illustration')
q2 = -0.05
q3 = pi/15
q5 = 0.1
q6 = pi/8
def InterpolationOfYDisplacement(alpha):
    N2, N3, N5, N6 = beamIllustration.GetInterpolationCoefficientsForYDisplacement(alpha)
    return N2 * q2 + N3 * q3 + N5 * q5 + N6 * q6
alpha = np.linspace(0, 1, 1000)
yInter = [InterpolationOfYDisplacement(i) for i in alpha]
figInter, axInter = plt.subplots(1, 1, figsize=[textWidth, 2.5])
axInter.plot(alpha, yInter)
axInter.set_xlabel(r'relative position $\alpha$ along element')
axInter.set_ylabel('interpolated y-displacement in m')
axInter.set_xlim([-0.1, 1.1])
axInter.set_ylim([-0.1, 0.15])
axInter.grid()
SaveFig(figInter, figPath, 'M_yDisplacementInterpolation', closeFig=True, saveFig=False)

#%% plot snapshots for three sliding joint positions for illustration (graphical abstract)
PrintHeadlineToConsole(text='Plot Snapshots for Illustration')
eigenmodesLiftedAndMassNormalizedListIllustration = beamIllustration.GetGlobalMassNormalizedEigenmodesListForNumberOfConstrainedNodes(nNodesConstrained=3, nEigenmodes=3)
figEigenmodes = beamIllustration.PlotEigenmodesSets(eigenmodesLiftedAndMassNormalizedListIllustration, nEigenmodesToPlot=3,
                                                    nodeNumbersConstrainedList=beamIllustration.nodeNumbersConstrainedList,
                                                    figSize=[textWidth, 4])
SaveFig(figEigenmodes, figPath, 'M_exampleInterpolations', closeFig=True, saveFig=False)

#%% define and plot the two test cases used: SIMPLE and EVOLVED
PrintHeadlineToConsole(text='Plot Test Cases')
startPosition = length/4        # [m] start position of slider (slider cannot move further to start)
stopPosition = length*(3/4)     # [m] stop position of slider (slider cannot move further to end)
startTime = tExperiment/4       # [s] start of test (slider starts, load gets varied)
stopTime = tExperiment*(3/4)    # [s] end of test (slider stops, load gets constant)
A = 50                          # [N] max. amplitude of load functions
def LoadTipFunctionSimple(mbs, t, load):
    return -A
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
def SlidingJointFunctionSimple(t, startPosition=startPosition, stopPosition=stopPosition,
                               startTime=startTime, stopTime=stopTime):
    if t < startTime:
        return startPosition
    if t > stopTime:
        return stopPosition
    else:
        return startPosition + ((stopPosition-startPosition)/(stopTime-startTime)) * (t-startTime)
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
# plot the two test cases
colorSlider = 'blue'
colorLoad = 'red'
figTestCases, axTestCases = plt.subplots(2, 1, figsize=[textWidth, 2.5], sharex=True)
time = np.linspace(0, tEnd, 5001)
# simple test case
axTestCases[0].plot(time, [SlidingJointFunctionSimple(t) for t in time], '*-', color=colorSlider, markevery=500)
axTestCases[0].tick_params(axis='y', labelcolor=colorSlider)
axTestCaseLoadSimple = axTestCases[0].twinx()
axTestCaseLoadSimple.plot(time, [LoadTipFunctionSimple(None, t, None) for t in time], color=colorLoad, alpha=0.75)
axTestCaseLoadSimple.tick_params(axis='y', labelcolor=colorLoad)
axTestCaseLoadSimple.set_ylim([-100, 100])
axTestCaseLoadSimple.set_yticks([-50, 0, 50])
# draw load behind slider
axTestCaseLoadSimple.set_zorder(1)
axTestCases[0].set_zorder(2)
axTestCases[0].patch.set_visible(False)
# complex test case
axTestCases[1].plot(time, [SlidingJointFunctionEvolved(t) for t in time], '*-', color=colorSlider, markevery=500)
axTestCases[1].tick_params(axis='y', labelcolor=colorSlider)
axTestCaseLoadEvolved = axTestCases[1].twinx()
axTestCaseLoadEvolved.plot(time, [LoadTipFunctionEvolved(None, t, None) for t in time], color=colorLoad, alpha=0.75)
axTestCaseLoadEvolved.tick_params(axis='y', labelcolor=colorLoad)
axTestCaseLoadEvolved.set_ylim([-100, 100])
axTestCaseLoadEvolved.set_yticks([-50, 0, 50])
# draw load behind slider
axTestCaseLoadEvolved.set_zorder(1)
axTestCases[1].set_zorder(2)
axTestCases[1].patch.set_visible(False)
# shared x and y labels
axTestCases[1].set_xlabel(timeStr)
axTestCases[1].set_xticks(np.arange(0, 16, 1))
figTestCases.supylabel(r'$\ast$ slider position $s(t)$ in m', color=colorSlider, x=0.045)
figTestCases.text(0.97, 0.5, r'tip load $F\,(t)$ in N', color=colorLoad, 
                  va='center', ha='right', rotation=90)
# add grids, shaded area, ...
for i, ax in enumerate(axTestCases):
    ax.grid()
    if i == 0:
        ax.axvspan(startTime, stopTime, color='gray', alpha=0.2, label=slidingStr, zorder=0)
        ax.legend(loc='upper left')
    else:
        ax.axvspan(startTime, stopTime, color='gray', alpha=0.2, zorder=0)
figTestCases.tight_layout(rect=[0, 0, 0.96, 1])
SaveFig(figTestCases, figPath, 'E_testCases', closeFig=False)

#%% do convergence analysis to get good amount of ANCF and beam elements
PrintHeadlineToConsole(text='Convergence Analysis')
if not onlyLoadResults:
    # set load and slidingJoint function for convergence analysis
    LoadTipFunctionConvergenceAnalysis = LoadTipFunctionEvolved
    SlidingJointFunctionConvergenceAnalysis = SlidingJointFunctionEvolved
    
    #++++++++++++++++++++++++++++++++++++++++++++++++++
    # initialize beam (to inherit some data to ANCF beam)
    beamConvergenceAnalysisNR = EulerBernulliBeam(beamTemplate=beamTemplateRubber)
    
    #++++++++++++++++++++++++++++++++++++++++++++++++++
    # ANCF elements
    nElementsANCFList = np.linspace(2, 44, 22, dtype=int)
    tipDisplacementConvergenceAnalysisANCFList = []
    startTime = tm.time()
    nCompletedAnalyses = 0
    nAnalyses= len(nElementsANCFList)
    for i, nElementsANCF in enumerate(nElementsANCFList):
        # retreive reference solution via ANCF beam with sliding joint
        SC = exu.SystemContainer()
        mbs = SC.AddSystem() # always create new mbs (or mbs.Reset())
        nGround = mbs.AddNode(NodePointGround())
        # this is needed to constrain ANCF beam on the 'left' side
        mCoordGround = mbs.AddMarker(MarkerNodeCoordinate(nodeNumber=nGround, coordinate=0,
                                                          visualization=VMarkerNodeCoordinates(show=False)))
        sensorOutputVariableTypeConvergenceAnalysisANCFList = [exu.OutputVariableType.Displacement] # here, we only need displacement
        sDictConvergenceAnalysisANCF = AddANCFBeamWithSlidingJointToMBSAndAddSensors(mbs, SlidingJointFunctionConvergenceAnalysis, LoadTipFunctionConvergenceAnalysis, length=length, rhoA=beamConvergenceAnalysisNR.rhoA, EI=beamConvergenceAnalysisNR.EI, EA=beamConvergenceAnalysisNR.EA,
                                                                                     nElementsANCF=nElementsANCF, mCoordGround=mCoordGround, sensorOutputVariableTypeANCFList=sensorOutputVariableTypeConvergenceAnalysisANCFList,
                                                                                     beamNodeReferencePositionList=beamConvergenceAnalysisNR.nodeReferencePositionsList)
        mbs.Assemble()
        simulationSettings = GetStandardSimulationAndSetVisualizationSettings(tEnd, h, exportFrames=False)
        mbs.SolveDynamic(simulationSettings=simulationSettings)
        outputDictConvergenceAnalysisANCF = GetOutputDictANCFFromSDictANCF(mbs, sDictANCF=sDictConvergenceAnalysisANCF, sensorOutputVariableTypeANCFList=sensorOutputVariableTypeConvergenceAnalysisANCFList)
        tipDisplacementConvergenceAnalysisANCFList += [outputDictConvergenceAnalysisANCF['OutputVariableType.Displacement'][:, -2]]
        nCompletedAnalyses += 1
        ProgressTracker(startTime, nAnalyses, nCompletedAnalyses, showETA=False)
    errorConvergenceAnalysisANCFList = []
    for tipDisplacementConvergenceAnalysisANCF in tipDisplacementConvergenceAnalysisANCFList[:-1]:
        errorConvergenceAnalysisANCFList += [AbsError(tipDisplacementConvergenceAnalysisANCF, tipDisplacementConvergenceAnalysisANCFList[-1])]
    errorConvergenceAnalysisANCFList += [0]
    AppendResultsToPKLFile(resultsFileName, 'convergenceAnalysis', {'nElementsANCFList': nElementsANCFList,
                                                                    'errorConvergenceAnalysisANCFList': errorConvergenceAnalysisANCFList})
    
    #++++++++++++++++++++++++++++++++++++++++++++++++++
    # beam elements
    nElementsNRList = np.linspace(2, 60, 30, dtype=int)
    tipDisplacementConvergenceAnalysisNRList = []
    startTime = tm.time()
    nCompletedAnalyses = 0
    nAnalyses= len(nElementsNRList)
    for i, nElementsBeam in enumerate(nElementsNRList):
        # initialize beam
        SC = exu.SystemContainer()
        mbs = SC.AddSystem() # always create new mbs (or mbs.Reset())
        nGround = mbs.AddNode(NodePointGround())
        mCoordsGround = mbs.AddMarker(MarkerNodeCoordinates(nodeNumber=nGround, visualization=VMarkerNodeCoordinates(show=False)))
        beamConvergenceAnalysisNR = EulerBernulliBeam(beamTemplate=beamTemplateRubber,
                                                      nElements=nElementsBeam, # OVERWRITE!
                                                      slidingJointFunction=SlidingJointFunctionConvergenceAnalysis,
                                                      loadTipFunction=LoadTipFunctionConvergenceAnalysis)
        
        #++++++++++++++++++++++++++++++++++++++++++++++++++
        # add beam to mbs
        sCoordinatesConvergenceAnalysisNR, sCoordinates_tConvergenceAnalysisNR = beamConvergenceAnalysisNR.AddBeamToMBSAndAddSensors(mbs, mCoordsGround=mCoordsGround, show=False)
        
        #++++++++++++++++++++++++++++++++++++++++++++++++++
        # simulate dynamically (WYSWYS)
        mbs.Assemble()
        simulationSettings = GetStandardSimulationAndSetVisualizationSettings(tEnd, h, exportFrames=False)
        mbs.SolveDynamic(simulationSettings=simulationSettings)
        
        #++++++++++++++++++++++++++++++++++++++++++++++++++
        # get results
        _, time, qConvergenceAnalysisNR, q_tConvergenceAnalysisNR = beamConvergenceAnalysisNR.GetTimeAndCoordinates(mbs, sCoordinatesConvergenceAnalysisNR, sCoordinates_tConvergenceAnalysisNR)
        tipDisplacementConvergenceAnalysisNRList += [qConvergenceAnalysisNR[:, -2]]
        nCompletedAnalyses += 1
        ProgressTracker(startTime, nAnalyses, nCompletedAnalyses, showETA=False)
    errorConvergenceAnalysisNRList = []
    for tipDisplacementConvergenceAnalysisNR in tipDisplacementConvergenceAnalysisNRList[:-1]:
        errorConvergenceAnalysisNRList += [AbsError(tipDisplacementConvergenceAnalysisNR, tipDisplacementConvergenceAnalysisNRList[-1])]
    errorConvergenceAnalysisNRList += [0]
    AppendResultsToPKLFile(resultsFileName, 'convergenceAnalysis', {'nElementsNRList': nElementsNRList,
                                                                    'errorConvergenceAnalysisNRList': errorConvergenceAnalysisNRList})
    
    #++++++++++++++++++++++++++++++++++++++++++++++++++
    # for debugging: plot displacement of tip
    if False:
        figDisplacementForDebugging, axDisplacementForDebugging = plt.subplots(1, 1, figsize=[textWidth, 3])
        for i in range(len(nElementsNRList)):
            axDisplacementForDebugging.plot(time, tipDisplacementConvergenceAnalysisANCFList[i], label='ANCF ele: ' + str(nElementsANCFList[i]))
            axDisplacementForDebugging.plot(time, tipDisplacementConvergenceAnalysisNRList[i], label='Beam ele: ' + str(nElementsNRList[i]))
        axDisplacementForDebugging.legend()

#++++++++++++++++++++++++++++++++++++++++++++++++++
# plot the convergence analysis
resultsDictConvergenceAnalysis = GetResultsDictFromPKLFile(resultsFileName, 'convergenceAnalysis')
nElementsANCFList = resultsDictConvergenceAnalysis['nElementsANCFList']
nDOFsANCFList = (nElementsANCFList + 1) * 4 # 4 DOfs per node
nElementsNRList = resultsDictConvergenceAnalysis['nElementsNRList']
nDOFsNRList = (nElementsNRList + 1) * 3 # 3 DOfs per node
errorConvergenceAnalysisANCFList = resultsDictConvergenceAnalysis['errorConvergenceAnalysisANCFList']
errorConvergenceAnalysisNRList = resultsDictConvergenceAnalysis['errorConvergenceAnalysisNRList']
figConvergenceAnalysis, axConvergenceAnalysis = plt.subplots(1, 1, figsize=[textWidth, 2.5])
axConvergenceAnalysis.loglog(nDOFsANCFList[:-1], errorConvergenceAnalysisANCFList[:-1], '-o', label=labelList[0], color=standardColors[0])
axConvergenceAnalysis.loglog(nDOFsNRList[:-1], errorConvergenceAnalysisNRList[:-1], '-*', label=labelList[1], color=standardColors[1])
axConvergenceAnalysis.grid()
axConvergenceAnalysis.legend()
axConvergenceAnalysis.set_xlabel('number of DOFs')
axConvergenceAnalysis.set_ylabel('abs. error transversal tip displacement in m')
SaveFig(figConvergenceAnalysis, figPath, 'E_convergenceAnalysis', closeFig=False)

#%% compute displacement error of globally reduced beam vs. non-reduced beam to obtain reduction error
PrintHeadlineToConsole(text='Reduction Error')
if not onlyLoadResults:
    # determine combinations that should be tested
    nSnapshotsList = np.arange(1, 30+1, 1)
    nEigenmodesPerSnapshotList = np.arange(1, 16+1, 1)
    nMinGlobalModes = 3
    nMaxGlobalModes = 120
    nGlobalModesEvaluationRange = [16, 20]
    nGlobalModesEvaluation = nGlobalModesEvaluationRange[1] - nGlobalModesEvaluationRange[0] + 1
    pairsToInvestigateList = []
    nGlobalModesToComputePerPairList = []
    nAnalyses = 0
    for nSnapshots in nSnapshotsList:
        for nEigenmodesPerSnapshot in nEigenmodesPerSnapshotList:
            product = nSnapshots * nEigenmodesPerSnapshot
            if product >= nMinGlobalModes and product <= nMaxGlobalModes:
                pairsToInvestigateList += [[nSnapshots, nEigenmodesPerSnapshot]]
                if product >= nGlobalModesEvaluationRange[1]:
                    nGlobalModesToComputePerPairList += [np.arange(nGlobalModesEvaluationRange[0], nGlobalModesEvaluationRange[1]+1, 1)]
                elif product >= nMinGlobalModes + nGlobalModesEvaluation:
                    nGlobalModesToComputePerPairList += [np.arange(product-nGlobalModesEvaluation+1, product+1, 1)]
                elif product >= nMinGlobalModes + 1:
                    nGlobalModesToComputePerPairList += [np.arange(nMinGlobalModes, product+1, 1)]
                elif product == nMinGlobalModes:
                    nGlobalModesToComputePerPairList += [np.array([product])]
                else:
                    print('WARNING: Reduction error computation setup failed!')
                nAnalyses += len(nGlobalModesToComputePerPairList[-1])
    print('Starting a total of ' + str(nAnalyses) + ' analyses.')
    
    #++++++++++++++++++++++++++++++++++++++++++++++++++
    # offline compute reference coordinates
    SC = exu.SystemContainer()
    mbs = SC.AddSystem() # always create new mbs (or mbs.Reset())
    nGround = mbs.AddNode(NodePointGround())
    mCoordsGround = mbs.AddMarker(MarkerNodeCoordinates(nodeNumber=nGround, visualization=VMarkerNodeCoordinates(show=False)))
    beamReductionErrorNR = EulerBernulliBeam(beamTemplate=beamTemplateRubber,
                                             slidingJointFunction=SlidingJointFunctionEvolved,
                                             loadTipFunction=LoadTipFunctionEvolved)
    sCoordinatesReductionErrorNR, sCoordinates_tReductionErrorNR = beamReductionErrorNR.AddBeamToMBSAndAddSensors(mbs, mCoordsGround=mCoordsGround, show=False)
    mbs.Assemble()
    simulationSettings = GetStandardSimulationAndSetVisualizationSettings(tEnd, h, exportFrames=False)
    mbs.SolveDynamic(simulationSettings=simulationSettings)
    _, time, qReductionErrorNR, q_tReductionErrorNR = beamReductionErrorNR.GetTimeAndCoordinates(mbs, sCoordinatesReductionErrorNR, sCoordinates_tReductionErrorNR)
    
    #++++++++++++++++++++++++++++++++++++++++++++++++++
    # now process the pairs that should be investigated
    nCompletedAnalyses = 0
    tipDisplacementErrorReductionErrorList = []
    tipMeanDisplacementErrorReductionErrorList = []
    beamDisplacementNormalizedErrorReductionErrorList = []
    beamMeanDispacementNormalizedErrorReductionErrorList = []
    startTime = tm.time()
    for i, pair in enumerate(pairsToInvestigateList):
        nSnapshots = pair[0]
        nEigenmodesPerSnapshot = pair[1]
        tipDisplacementError = []
        beamDisplacementNormalizedError = []
        for nGlobalModes in nGlobalModesToComputePerPairList[i]:
            # initialize beam
            SC = exu.SystemContainer()
            mbs = SC.AddSystem() # always create new mbs (or mbs.Reset())
            nGround = mbs.AddNode(NodePointGround())
            mCoordsGround = mbs.AddMarker(MarkerNodeCoordinates(nodeNumber=nGround, visualization=VMarkerNodeCoordinates(show=False)))
            beamReductionErrorPMR = EulerBernulliBeam(beamTemplate=beamTemplateRubber,
                                                      slidingJointFunction=SlidingJointFunctionEvolved,
                                                      loadTipFunction=LoadTipFunctionEvolved)
            
            #++++++++++++++++++++++++++++++++++++++++++++++++++
            # GLOBALLY reduced beam and add it
            # we start always with the full globalReductionBasisInitial
            beamReductionErrorPMR.GlobalReduceBeamWrapper(nSnapshots=nSnapshots, nEigenmodesPerSnapshot=nEigenmodesPerSnapshot,
                                                          nGlobalModes=nGlobalModes)
            sCoordinatesReductionErrorPMR, sCoordinates_tReductionErrorPMR = beamReductionErrorPMR.AddBeamToMBSAndAddSensors(mbs, mCoordsGround=mCoordsGround, show=False)
            
            #++++++++++++++++++++++++++++++++++++++++++++++++++
            # simulate dynamically (WYSWYS)
            mbs.Assemble()
            simulationSettings = GetStandardSimulationAndSetVisualizationSettings(tEnd, h, exportFrames=False)
            mbs.SolveDynamic(simulationSettings=simulationSettings)
            
            #++++++++++++++++++++++++++++++++++++++++++++++++++
            # get results
            _, time, qReductionErrorPMR, q_tReductionErrorPMR = beamReductionErrorPMR.GetTimeAndCoordinates(mbs, sCoordinatesReductionErrorPMR, sCoordinates_tReductionErrorPMR)
            tipDisplacementError += [AbsError(qReductionErrorPMR[:, -2], qReductionErrorNR[:, -2])]
            beamDisplacementNormalizedError += [NRMSEVector(qReductionErrorPMR[:, 1::beamReductionErrorPMR.nDOFsPerNode], qReductionErrorNR[:, 1::beamReductionErrorNR.nDOFsPerNode])]
            nCompletedAnalyses += 1
            ProgressTracker(startTime, nAnalyses, nCompletedAnalyses)
        tipDisplacementErrorReductionErrorList += [tipDisplacementError]
        tipMeanDisplacementErrorReductionErrorList += [np.mean(tipDisplacementError)]
        beamDisplacementNormalizedErrorReductionErrorList += [beamDisplacementNormalizedError]
        beamMeanDispacementNormalizedErrorReductionErrorList += [np.mean(beamDisplacementNormalizedError)]
    AppendResultsToPKLFile(resultsFileName, 'reductionError', {'nSnapshotsList':nSnapshotsList,
                                                               'nEigenmodesPerSnapshotList':nEigenmodesPerSnapshotList,
                                                               'pairsToInvestigateList': pairsToInvestigateList,
                                                               'nGlobalModesToComputePerPairList': nGlobalModesToComputePerPairList,
                                                               'tipDisplacementErrorReductionErrorList': tipDisplacementErrorReductionErrorList,
                                                               'tipMeanDisplacementErrorReductionErrorList':tipMeanDisplacementErrorReductionErrorList,
                                                               'beamDisplacementNormalizedErrorReductionErrorList': beamDisplacementNormalizedErrorReductionErrorList,
                                                               'beamMeanDispacementNormalizedErrorReductionErrorList':beamMeanDispacementNormalizedErrorReductionErrorList,
                                                               'qReductionErrorNR': qReductionErrorNR})

#++++++++++++++++++++++++++++++++++++++++++++++++++
# plot the reduction error
resultsDictReductionError = GetResultsDictFromPKLFile(resultsFileName, 'reductionError')
nSnapshotsList = resultsDictReductionError['nSnapshotsList']
nEigenmodesPerSnapshotList = resultsDictReductionError['nEigenmodesPerSnapshotList']
pairsToInvestigateList = resultsDictReductionError['pairsToInvestigateList']
tipMeanDisplacementErrorReductionErrorList = resultsDictReductionError['tipMeanDisplacementErrorReductionErrorList']
beamMeanDispacementNormalizedErrorReductionErrorList = resultsDictReductionError['beamMeanDispacementNormalizedErrorReductionErrorList']
tipMeanDisplacementErrorMatrix = np.full([len(nEigenmodesPerSnapshotList), len(nSnapshotsList)], np.nan)
beamMeanDisplacementErrorMatrix = np.full([len(nEigenmodesPerSnapshotList), len(nSnapshotsList)], np.nan)
for i, pair in enumerate(pairsToInvestigateList):
    nSnapshots = pair[0]
    nEigenmodesPerSnapshot = pair[1]
    idxSnapshot = np.where(nSnapshotsList == nSnapshots)
    idxEigenmodesPerSnapshot = np.where(nEigenmodesPerSnapshotList == nEigenmodesPerSnapshot)
    tipMeanDisplacementErrorMatrix[idxEigenmodesPerSnapshot, idxSnapshot] = tipMeanDisplacementErrorReductionErrorList[i]
    beamMeanDisplacementErrorMatrix[idxEigenmodesPerSnapshot, idxSnapshot] = beamMeanDispacementNormalizedErrorReductionErrorList[i]
figReductionError, axReductionError = plt.subplots(1, 1, figsize=[textWidth, 3.5])
# im = axReductionError.imshow(tipMeanDisplacementErrorMatrix, cmap='RdYlGn_r')
im = axReductionError.imshow(beamMeanDisplacementErrorMatrix, cmap='RdYlGn_r', vmin=0, vmax=1)
axReductionError.set_xticks(nSnapshotsList - 1)
axReductionError.set_xticklabels(nSnapshotsList)
axReductionError.set_yticks(nEigenmodesPerSnapshotList - 1)
axReductionError.set_yticklabels(nEigenmodesPerSnapshotList)
axReductionError.invert_yaxis()
axReductionError.set_xlabel(r'number $n_\mathrm{snap.}$ of snapshots')
axReductionError.set_ylabel(r'number $r_\mathrm{snap.}$ of eigenmodes per snapshot')
plt.colorbar(im, label='normalized error beam y-displacement')
SaveFig(figReductionError, figPath, 'E_normalizedBeamDisplacementHeatmap', closeFig=False)

#++++++++++++++++++++++++++++++++++++++++++++++++++
# get best combination for treated problem
bestPairIndices = np.unravel_index(np.nanargmin(beamMeanDisplacementErrorMatrix), beamMeanDisplacementErrorMatrix.shape)
nSnapshotsBest = nSnapshotsList[bestPairIndices[1]]
nEigenmodesPerSnapshotBest = nEigenmodesPerSnapshotList[bestPairIndices[0]]
print('Best pair: nSapshots = ' + str(nSnapshotsBest) + '; nEigenmodesPerSnapshot = ' + str(nEigenmodesPerSnapshotBest))

# #%% compute strain energy error
# PrintHeadlineToConsole(text='Strain Energy Error')
# if not onlyLoadResults:
#     #%%
#     # offline compute reference strain energy sum (non-reduced beam)
#     SC = exu.SystemContainer()
#     mbs = SC.AddSystem() # always create new mbs (or mbs.Reset())
#     nGround = mbs.AddNode(NodePointGround())
#     mCoordsGround = mbs.AddMarker(MarkerNodeCoordinates(nodeNumber=nGround, visualization=VMarkerNodeCoordinates(show=False)))
#     beamStrainEnergySumNR = EulerBernulliBeam(beamTemplate=beamTemplateRubber,
#                                               slidingJointFunction=SlidingJointFunctionEvolved,
#                                               loadTipFunction=LoadTipFunctionEvolved)
#     sCoordinatesStrainEnergySumBeamNR, sCoordinates_tStrainEnergySumBeamNR = beamStrainEnergySumNR.AddBeamToMBSAndAddSensors(mbs, mCoordsGround=mCoordsGround, show=False)
#     mbs.Assemble()
#     simulationSettings = GetStandardSimulationAndSetVisualizationSettings(tEnd, h, exportFrames=False)
#     mbs.SolveDynamic(simulationSettings=simulationSettings)
#     _, time, qStrainEnergySumBeamNR, q_tStrainEnergySumBeamNR = beamStrainEnergySumNR.GetTimeAndCoordinates(mbs, sCoordinatesStrainEnergySumBeamNR, sCoordinates_tStrainEnergySumBeamNR)
#     strainEnergySumNR = beamStrainEnergySumNR.GetSumOfBeamStrainEnergy(qStrainEnergySumBeamNR)
#     print('Non-reduced strain energy sum:', strainEnergySumNR, 'J.')
    
   
    
#     #%%
    
#     #++++++++++++++++++++++++++++++++++++++++++++++++++
#     # determine combinations that should be tested
#     nGlobalModesMaxPossible = 50
#     nGlobalModesMinPossible = 4
#     nGlobalModesEvaluationRange = [15, 20]
#     pairsToBeInvestigatedList = []
#     for nSnap in range(1, nGlobalModesMaxPossible + 1):
#         for rSnap in range(1, nGlobalModesMaxPossible + 1):
#             product = nSnap * rSnap
#             if nGlobalModesMinPossible <= product <= nGlobalModesMaxPossible:
#                 pairsToBeInvestigatedList += [(nSnap, rSnap)]
#     # pairs that should be investigated (first number is nSnap, second is rSnap)
#     pairsToBeInvestigatedList = sorted(list(set(pairsToBeInvestigatedList)))
#     nPairsToBeInvestigated = len(pairsToBeInvestigatedList)
#     print('Total number of combinations to investigate:', nPairsToBeInvestigated)
#     # now check how many analyses are done and prepare variation
#     nAnalyses = 0
#     nGlobalModesEvaluatedList = []
#     for pair in pairsToBeInvestigatedList:
#         nGlobalModesPossible = pair[0] * pair[1]
#         if nGlobalModesPossible <= nGlobalModesEvaluationRange[1]:
#             nGlobalModesEvaluationStart = nGlobalModesPossible
#         else:
#             nGlobalModesEvaluationStart = nGlobalModesEvaluationRange[1]
#         nGlobalModesEvaluatedPerPairList = []
#         for i in range(nGlobalModesEvaluationRange[1]-nGlobalModesEvaluationRange[0]+1):
#             nGlobalModesEvaluation = nGlobalModesEvaluationStart - i
#             if nGlobalModesEvaluation > 2:
#                 nGlobalModesEvaluatedPerPairList += [nGlobalModesEvaluation]
#         nGlobalModesEvaluatedList += [nGlobalModesEvaluatedPerPairList]
#         nAnalyses += len(nGlobalModesEvaluatedList[-1])
#     print('Total number of analyses:', nAnalyses)
    
#     #++++++++++++++++++++++++++++++++++++++++++++++++++
#     # do MAIN variation
#     strainEnergySumList = []
#     nAnalysesCompleted = 0
#     startTime = tm.time()
#     for i, pair in enumerate(pairsToBeInvestigatedList):
#         # OFFLINE compute initial global reduction basis
#         beamStrainEnergyPMR = EulerBernulliBeam(beamTemplate=beamTemplateRubber)
#         eigenmodesLiftedAndMassNormalizedList = beamStrainEnergyPMR.GetGlobalMassNormalizedEigenmodesListForNumberOfConstrainedNodes(nNodesConstrained=pair[0],
#                                                                                                                                      nEigenmodes=pair[1],
#                                                                                                                                      verbose=True)
#         globalReductionBasisInitial = beamStrainEnergyPMR.GetGlobalReductionBasisUsingEigenmodesList(eigenmodesList=eigenmodesLiftedAndMassNormalizedList,
#                                                                                                      nGlobalModes=nGlobalModesEvaluatedList[i][0], verbose=True)
        
#         #++++++++++++++++++++++++++++++++++++++++++++++++++
#         # iterate over the global modes and remove them successively
#         strainEnergySumForPairList = []
#         for nGlobalModesUsed in nGlobalModesEvaluatedList[i]:
#             # initialize beam
#             SC = exu.SystemContainer()
#             mbs = SC.AddSystem() # always create new mbs (or mbs.Reset())
#             nGround = mbs.AddNode(NodePointGround())
#             mCoordsGround = mbs.AddMarker(MarkerNodeCoordinates(nodeNumber=nGround, visualization=VMarkerNodeCoordinates(show=False)))
#             beamStrainEnergySumPMR = EulerBernulliBeam(beamTemplate=beamTemplateRubber,
#                                                        slidingJointFunction=SlidingJointFunctionEvolved,
#                                                        loadTipFunction=LoadTipFunctionEvolved)
            
#             #++++++++++++++++++++++++++++++++++++++++++++++++++
#             # GLOBALLY reduced beam and add it (here we can not use the wrapper for global reduction)
#             # we start always with the full globalReductionBasisInitial
#             beamStrainEnergySumPMR.ReduceBeam(reductionBasis=globalReductionBasisInitial[:, 0:nGlobalModesUsed])
#             sCoordinatesStrainEnergySumBeamPMR, sCoordinates_tStrainEnergySumBeamPMR = beamStrainEnergySumPMR.AddBeamToMBSAndAddSensors(mbs, mCoordsGround=mCoordsGround, show=False)
            
#             #++++++++++++++++++++++++++++++++++++++++++++++++++
#             # simulate dynamically (WYSWYS)
#             mbs.Assemble()
#             simulationSettings = GetStandardSimulationAndSetVisualizationSettings(tEnd, h, exportFrames=False)
#             mbs.SolveDynamic(simulationSettings=simulationSettings)
            
#             #++++++++++++++++++++++++++++++++++++++++++++++++++
#             # get results
#             _, time, qStrainEnergySumBeamPMR, q_tStrainEnergySumBeamPMR = beamStrainEnergySumPMR.GetTimeAndCoordinates(mbs, sCoordinatesStrainEnergySumBeamPMR, sCoordinates_tStrainEnergySumBeamPMR)
#             strainEnergySum = beamStrainEnergySumPMR.GetSumOfBeamStrainEnergy(qStrainEnergySumBeamPMR)
#             strainEnergySumForPairList += [strainEnergySum] # this varies in length
            
            
            
#             nAnalysesCompleted += 1
            
#             #++++++++++++++++++++++++++++++++++++++++++++++++++
#             # track progress
#             neededTime = (tm.time() - startTime) / nAnalysesCompleted
#             nAnalysesToGo = nAnalyses - nAnalysesCompleted
#             timeToGo = nAnalysesToGo * neededTime
#             hoursToGo = int(timeToGo) // 3600
#             minutesToGo = (int(timeToGo) % 3600) // 60
#             secondsToGo = int(timeToGo) % 60
#             print(f'Completed {(nAnalysesCompleted/nAnalyses) * 100:05.1f}%. ETA: {hoursToGo:02d}:{minutesToGo:02d}:{secondsToGo:02d}')
#         strainEnergySumList += [strainEnergySumForPairList]
#     AppendResultsToPKLFile(resultsFileName, 'strainEnergySum', {'pairsToBeInvestigatedList': pairsToBeInvestigatedList,
#                                                                 'nGlobalModesEvaluatedList': nGlobalModesEvaluatedList,
#                                                                 'strainEnergySumNR': strainEnergySumNR,
#                                                                 'strainEnergySumList': strainEnergySumList})
    
#     #%% 
#     # compute some specific pairs to the fullest for visualization and closer inspection
#     nGlobalModesEvaluatedSpecificList = [[i for i in np.flipud(np.arange(3, 23))],
#                                          [i for i in np.flipud(np.arange(3, 23))],
#                                          [i for i in np.flipud(np.arange(3, 9))]]
#     strainEnergySumSpecificList = GetStrainEnergySumOverSimulation(beamTemplate=beamTemplateRubber,
#                                                                    slidingJointFunction=SlidingJointFunctionEvolved,
#                                                                    loadTipFunction=LoadTipFunctionEvolved,
#                                                                    pairsToBeInvestigatedList=[(6, 6), (2, 12), (8, 1)],
#                                                                    nGlobalModesEvaluatedList=nGlobalModesEvaluatedSpecificList,
#                                                                    tEnd=tEnd, h=h)
    
    
#     #%%
# #++++++++++++++++++++++++++++++++++++++++++++++++++
# # compute and plot the errors
# resultsDictStrainEnergySum = GetResultsDictFromPKLFile(resultsFileName, 'strainEnergySum')
# pairsToBeInvestigatedList = resultsDictStrainEnergySum['pairsToBeInvestigatedList']
# nGlobalModesEvaluatedList = resultsDictStrainEnergySum['nGlobalModesEvaluatedList']
# strainEnergySumNR = resultsDictStrainEnergySum['strainEnergySumNR']
# strainEnergySumList = resultsDictStrainEnergySum['strainEnergySumList']
# nPairsToBeInvestigated = len(pairsToBeInvestigatedList)
# # compute the average strain energy sum error for each pair
# averageStrainEnergySumError = []
# for i in range(nPairsToBeInvestigated):
#     strainEnergySumErrorList = []
#     for strainEnergySum in strainEnergySumList[i]:
#         strainEnergySumErrorList += [np.abs(strainEnergySumNR-strainEnergySum)]
#     averageStrainEnergySumError += [np.mean(strainEnergySumErrorList)]
# # plot the error matrix
# nSnapshotsUniqueList = sorted(list(set([p[0] for p in pairsToBeInvestigatedList])))
# nEigenmodesPerSnapshotUniqueList = sorted(list(set([p[1] for p in pairsToBeInvestigatedList])))
# errorMatrix = np.full([len(nEigenmodesPerSnapshotUniqueList), len(nSnapshotsUniqueList)], np.nan)
# for (nSnap, rSnap), error in zip(pairsToBeInvestigatedList, averageStrainEnergySumError):
#     i = nEigenmodesPerSnapshotUniqueList.index(rSnap)
#     j = nSnapshotsUniqueList.index(nSnap)
#     errorMatrix[i, j] = error
# # plot
# vmin = np.min(averageStrainEnergySumError)
# figHeatmapStrainErrorSum, axHeatmapStrainErrorSum = plt.subplots(figsize=[textWidth, 3.5])
# im = axHeatmapStrainErrorSum.imshow(errorMatrix, cmap='RdYlGn_r', aspect='auto',
#                                     vmin=vmin)
# axHeatmapStrainErrorSum.set_xticks(range(len(nSnapshotsUniqueList)))
# axHeatmapStrainErrorSum.set_xticklabels(nSnapshotsUniqueList)
# axHeatmapStrainErrorSum.set_yticks(range(len(nEigenmodesPerSnapshotUniqueList)))
# axHeatmapStrainErrorSum.set_yticklabels(nEigenmodesPerSnapshotUniqueList)
# axHeatmapStrainErrorSum.invert_yaxis() # this we need because of how a matrix is filled up ...
# axHeatmapStrainErrorSum.set_xlabel(r'number $n_\mathrm{snap.}$ of snapshots')
# axHeatmapStrainErrorSum.set_ylabel(r'number $r_\mathrm{snap.}$ of eigenmodes per snapshot')
# # axHeatmapStrainErrorSum.plot(np.arange(nSnapshotsUniqueList[0], nSnapshotsUniqueList[-1]), np.arange(nEigenmodesPerSnapshotUniqueList[0], nEigenmodesPerSnapshotUniqueList[-1]), color='black')
# axHeatmapStrainErrorSum.text(nSnapshotsUniqueList[-1]-7, nSnapshotsUniqueList[-1]-3, r'$r_\mathrm{snap.}=n_\mathrm{snap.}$', ha='left', va='bottom')
# plt.colorbar(im, label='average strain energy sum error in J')
# axHeatmapStrainErrorSum.set_axisbelow(True)
# SaveFig(figHeatmapStrainErrorSum, figPath, 'E_strainEnergyHeatmap', closeFig=False)


# #%%
# # for each pair a line, for each mode number a column
# strainEnergySumErrorMatrix = np.zeros([nPairsToBeInvestigated, nGlobalModesUsed])
# for i in range(nPairsToBeInvestigated):
#     for j in range(nGlobalModesUsed):
#         strainEnergySumErrorMatrix[i, j] = abs(strainEnergySumList[i][j]-strainEnergySumNR)
# # bar chart of three combinations to show filtering effect by Benner et. al
# # figBarChartStrainSumError, axBarChartStrainSumError = plt.subplots(1, 1, figsize=[textWidth, 2.5])
# # indexOfMaxNumberOfGlobalModesPlotted = 0 # we always start plotting from all nGlobalModesMaxTested modes
# # indexListForBarChart = []
# # indexListForBarChart += [pairsToBeInvestigatedList.index((6, 6))]
# # indexListForBarChart += [pairsToBeInvestigatedList.index((12, 2))]
# # indexListForBarChart += [pairsToBeInvestigatedList.index((2, 12))]
# # x = np.arange(nGlobalModesUsed-indexOfMaxNumberOfGlobalModesPlotted)
# # for i, index in enumerate(indexListForBarChart):
# #     nSnapshots = pairsToBeInvestigatedList[index][0]
# #     nEigenmodesPerSnapshot = pairsToBeInvestigatedList[index][1]
# #     axBarChartStrainSumError.plot(x, strainEnergySumErrorMatrix[index, indexOfMaxNumberOfGlobalModesPlotted:]/(1e3),
# #                                   markerList[i], label=r'$n_\mathrm{snap.}=' + str(nSnapshots) + r'$, $r_\mathrm{snap.}=' + str(nEigenmodesPerSnapshot) + '$',
# #                                   color=standardColors[i])
# #     if i == 1:
# #         axBarChartStrainSumError.fill_between(x, strainEnergySumErrorMatrix[index, indexOfMaxNumberOfGlobalModesPlotted:]/(1e3), alpha=0.2,
# #                                               color=standardColors[i])
# #         axBarChartStrainSumError.text(17.2, 0.6, 'cum. strain\n energy sum error', ha='center', va='center',
# #                                       color='black')
# # axBarChartStrainSumError.set_xticks(x, nGlobalModesUsedList[indexOfMaxNumberOfGlobalModesPlotted:])
# # axBarChartStrainSumError.set_xlabel(r'number $r_\mathrm{glob.}$ of global modes used')
# # axBarChartStrainSumError.set_ylabel('abs. strain energy sum error in kJ')
# # axBarChartStrainSumError.legend()
# # axBarChartStrainSumError.set_axisbelow(True)
# # axBarChartStrainSumError.grid()
# # SaveFig(figBarChartStrainSumError, figPath, 'E_strainEnergyBarChart', closeFig=False)
# # heatmap to show spatial coverage vs. mode richness per snapshot
# # for each pair, accumulate the strain energy error sum over ALL tested global mode numbers
# strainEnergySumSumErrorForPairList = []
# for i in range(nPairsToBeInvestigated):
#     strainEnergySumSumErrorForPairList += [np.sum(strainEnergySumErrorMatrix[i, :])]
# # which nSnap do we investigate and which rSnap?
# nSnapshotsUniqueList = sorted(list(set([p[0] for p in pairsToBeInvestigatedList])))
# nEigenmodesPerSnapshotUniqueList = sorted(list(set([p[1] for p in pairsToBeInvestigatedList])))
# # initialize the error mnatrix
# errorMatrix = np.full([len(nEigenmodesPerSnapshotUniqueList), len(nSnapshotsUniqueList)], np.nan)
# for (nSnap, rSnap), error in zip(pairsToBeInvestigatedList, strainEnergySumSumErrorForPairList):
#     i = nEigenmodesPerSnapshotUniqueList.index(rSnap)
#     j = nSnapshotsUniqueList.index(nSnap)
#     errorMatrix[i, j] = error
# # plot
# figHeatmapStrainErrorSum, axHeatmapStrainErrorSum = plt.subplots(figsize=[textWidth, 3.5])
# im = axHeatmapStrainErrorSum.imshow(errorMatrix, cmap='RdYlGn_r', aspect='auto')
# axHeatmapStrainErrorSum.set_xticks(range(len(nSnapshotsUniqueList)))
# axHeatmapStrainErrorSum.set_xticklabels(nSnapshotsUniqueList)
# axHeatmapStrainErrorSum.set_yticks(range(len(nEigenmodesPerSnapshotUniqueList)))
# axHeatmapStrainErrorSum.set_yticklabels(nEigenmodesPerSnapshotUniqueList)
# axHeatmapStrainErrorSum.invert_yaxis() # this we need because of how a matrix is filled up ...
# axHeatmapStrainErrorSum.set_xlabel(r'number $n_\mathrm{snap.}$ of snapshots')
# axHeatmapStrainErrorSum.set_ylabel(r'number $r_\mathrm{snap.}$ of eigenmodes per snapshot')
# # axHeatmapStrainErrorSum.plot(np.arange(nSnapshotsUniqueList[0], nSnapshotsUniqueList[-1]), np.arange(nEigenmodesPerSnapshotUniqueList[0], nEigenmodesPerSnapshotUniqueList[-1]), color='black')
# axHeatmapStrainErrorSum.text(nSnapshotsUniqueList[-1]-7, nSnapshotsUniqueList[-1]-3, r'$r_\mathrm{snap.}=n_\mathrm{snap.}$', ha='left', va='bottom')
# plt.colorbar(im, label='cum. abs. strain energy sum error in kJ')
# axHeatmapStrainErrorSum.set_axisbelow(True)
# SaveFig(figHeatmapStrainErrorSum, figPath, 'E_strainEnergyHeatmap', closeFig=False)

# #%% show the Hamiltonian of the simple sliding example
# PrintHeadlineToConsole(text='Hamiltonian Simple Example')
# if not onlyLoadResults:
#     # initialize beam (to inherit some data to ANCF beam)
#     beamBCImplementationNR = EulerBernulliBeam(beamTemplate=beamTemplateRubber)
    
#     #++++++++++++++++++++++++++++++++++++++++++++++++++
#     # add ANCF beam with sliding joint
#     SC = exu.SystemContainer()
#     mbs = SC.AddSystem() # always create new mbs (or mbs.Reset())
#     nGround = mbs.AddNode(NodePointGround())
#     # this is needed to constrain ANCF beam on the 'left' side
#     mCoordGround = mbs.AddMarker(MarkerNodeCoordinate(nodeNumber=nGround, coordinate=0,
#                                                       visualization=VMarkerNodeCoordinates(show=False)))
#     sensorOutputVariableTypeBCImplementationANCFList = [exu.OutputVariableType.Displacement,
#                                                         exu.OutputVariableType.Velocity,
#                                                         exu.OutputVariableType.CurvatureLocal,
#                                                         exu.OutputVariableType.TorqueLocal]
#     sDictBCImplementationANCF = AddANCFBeamWithSlidingJointToMBSAndAddSensors(mbs, SlidingJointFunctionSimple, LoadTipFunctionSimple, length=length, rhoA=beamBCImplementationNR.rhoA, EI=beamBCImplementationNR.EI, EA=beamBCImplementationNR.EA,
#                                                                               nElementsANCF=nElementsANCF, mCoordGround=mCoordGround, sensorOutputVariableTypeANCFList=sensorOutputVariableTypeBCImplementationANCFList,
#                                                                               beamNodeReferencePositionList=beamBCImplementationNR.nodeReferencePositionsList)
        
#     #++++++++++++++++++++++++++++++++++++++++++++++++++
#     # add beam NR
#     mCoordsGround = mbs.AddMarker(MarkerNodeCoordinates(nodeNumber=nGround, visualization=VMarkerNodeCoordinates(show=False)))
#     beamBCImplementationNR = EulerBernulliBeam(beamTemplate=beamTemplateRubber,
#                                                slidingJointFunction=SlidingJointFunctionSimple,
#                                                loadTipFunction=LoadTipFunctionSimple)
#     sCoordinatesBCImplementationNR, sCoordinates_tBCImplementationNR = beamBCImplementationNR.AddBeamToMBSAndAddSensors(mbs, mCoordsGround=mCoordsGround, show=False)
    
#     #++++++++++++++++++++++++++++++++++++++++++++++++++
#     # add beam PMR
#     beamBCImplementationPMR = EulerBernulliBeam(beamTemplate=beamTemplateRubber,
#                                                 slidingJointFunction=SlidingJointFunctionSimple,
#                                                 loadTipFunction=LoadTipFunctionSimple)
#     beamBCImplementationPMR.GlobalReduceBeamWrapper(nGlobalModes=18, nSnapshots=6, nEigenmodesPerSnapshot=6)
#     sCoordinatesBCImplementationPMR, sCoordinates_tBCImplementationPMR = beamBCImplementationPMR.AddBeamToMBSAndAddSensors(mbs, mCoordsGround=mCoordsGround, show=False)
    
#     #++++++++++++++++++++++++++++++++++++++++++++++++++
#     # simulate
#     mbs.Assemble()
#     simulationSettings = GetStandardSimulationAndSetVisualizationSettings(tEnd, h, exportFrames=False)
#     mbs.SolveDynamic(simulationSettings=simulationSettings)
    
#     #++++++++++++++++++++++++++++++++++++++++++++++++++
#     # get outputs
#     # ANCF
#     outputDictBCImplementationANCF = GetOutputDictANCFFromSDictANCF(mbs, sDictANCF=sDictBCImplementationANCF, sensorOutputVariableTypeANCFList=sensorOutputVariableTypeBCImplementationANCFList, verbose=True)
#     # NR
#     nTimeSteps, time, qBCimplementationNR, q_tBCImplementationNR = beamBCImplementationNR.GetTimeAndCoordinates(mbs, sCoordinatesBCImplementationNR, sCoordinates_tBCImplementationNR)
#     # PMR
#     _, _, qBCImplementationPMR, q_tBCImplementationPMR = beamBCImplementationPMR.GetTimeAndCoordinates(mbs, sCoordinatesBCImplementationPMR, sCoordinates_tBCImplementationPMR)
    
#     #++++++++++++++++++++++++++++++++++++++++++++++++++
#     # compute energies over time
#     EBCImplementationANCF = GetKineticAndStrainAndPotentialANCFEnergyFromOutputDictANCF(outputDictBCImplementationANCF, nTimeSteps=nTimeSteps, rhoA=beamBCImplementationNR.rhoA, LoadTipFunction=LoadTipFunctionSimple,
#                                                                                         length=length, nBeamElements=nElementsBeam, sensorsWritePeriod=simulationSettings.solutionSettings.sensorsWritePeriod)
#     # NR
#     EBCImplementationNR = beamBCImplementationNR.GetKineticAndStrainAndPotentialBeamEnergy(qBCimplementationNR, q_tBCImplementationNR, sensorWritePeriod=simulationSettings.solutionSettings.sensorsWritePeriod)
#     # PMR
#     EBCImplementationPMR = beamBCImplementationPMR.GetKineticAndStrainAndPotentialBeamEnergy(qBCImplementationPMR, q_tBCImplementationPMR, sensorWritePeriod=simulationSettings.solutionSettings.sensorsWritePeriod)
    
#     #%%
#     # plot
#     markerInterval = 150
#     figEnergyOverTime, axEnergyOverTime = plt.subplots(1, 1, figsize=[textWidth, 2.5])
#     markEvery = (0, markerInterval)
#     axEnergyOverTime.plot(time, EBCImplementationANCF, markerList[0], label=labelList[0], markevery=markEvery, color=standardColors[0])
#     markEvery = (int(markerInterval/3), markerInterval)
#     axEnergyOverTime.plot(time, EBCImplementationNR, markerList[1], label=labelList[1], markevery=markEvery, color=standardColors[1])
#     markEvery = (int(2*markerInterval/3), markerInterval)
#     axEnergyOverTime.plot(time, EBCImplementationPMR, markerList[2], label=labelList[2], markevery=markEvery, color=standardColors[2])
#     axEnergyOverTime.set_xlabel(timeStr)
#     axEnergyOverTime.set_ylabel('Hamiltonian of the system in J')
#     axEnergyOverTime.axvspan(startTime, stopTime, color='gray', alpha=0.2, label=slidingStr)
#     axEnergyOverTime.legend()
#     axEnergyOverTime.grid()
#     SaveFig(figEnergyOverTime, figPath, 'E_hamiltonian', closeFig=False)

# #%%
# beamStrainEnergyPMR = EulerBernulliBeam(beamTemplate=beamTemplateRubber)
# eigenmodesLiftedAndMassNormalizedList = beamStrainEnergyPMR.GetGlobalMassNormalizedEigenmodesListForNumberOfConstrainedNodes(nNodesConstrained=23,
#                                                                                                                              nEigenmodes=1,
#                                                                                                                              verbose=True)
# globalReductionBasisInitial = beamStrainEnergyPMR.GetGlobalReductionBasisUsingEigenmodesList(eigenmodesList=eigenmodesLiftedAndMassNormalizedList,
#                                                                                              nGlobalModes=18, verbose=True)
# figure = beamStrainEnergyPMR.PlotEigenmodesSets(eigenmodesList=[globalReductionBasisInitial], nodeNumbersConstrainedList=[],
#                                        nEigenmodesToPlot=4, easyPlotting=True)

# # #%%
# # qCenteredNR = qStrainEnergySumBeamNR - np.mean(qStrainEnergySumBeamNR, axis=0)
# # Q = np.fft.rfft(qCenteredNR, axis=0)
# # n = qStrainEnergySumBeamNR.shape[0]
# # freqs = np.fft.rfftfreq(n, d=simulationSettings.solutionSettings.sensorsWritePeriod)
# # amplitude = np.abs(Q) / n
# # amp_global = np.sqrt(np.mean(amplitude**2, axis=1))
# # plt.plot(freqs, amp_global)
# # plt.xlabel("Frequency [Hz]")
# # plt.ylabel("Amplitude")
# # plt.title("Global FFT of coordinates")
# # plt.grid()
# # plt.show()
