#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#
# Details:  
# Author:   Sebastian Weyrer
# Date:     2026-04-21
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import pickle
import numpy as np
import exudyn as exu
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from exudyn.utilities import *
from beams import EulerBernulliBeam, BeamTemplate
from utilities import *
from pathlib import Path
from datetime import datetime
print('EXUDYN says hello with version', exu.config.Version() + '.')

#++++++++++++++++++++++++++++++++++++++++++++++++++
# decide whether only to load results or produce them
onlyLoadResults = True

#++++++++++++++++++++++++++++++++++++++++++++++++++
# setup of 'test' beams with which the experiments are made (and ANCF reference settings)
length = 5          # [m] length of the beams
nElementsANCF = 12  # from convergence analysis (see later)
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
resultsFileName = 'results.pkl'
if not onlyLoadResults:
    print('Creating empty results file to save results.')
    data = {'metaInfo': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'results':{}}
    with open(resultsFileName, 'wb') as f:
        pickle.dump(data, f)
else:
    with open(resultsFileName, 'rb') as f:
        data = pickle.load(f)
    print(f"Loading existing results created on {data['metaInfo']}.")
    
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
figTestCases, axTestCases = plt.subplots(2, 1, figsize=[textWidth, 3], sharex=True)
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
figTestCases.supylabel(r'$\ast$ slider position in m', color=colorSlider, x=0.03)
figTestCases.text(0.97, 0.5, 'tip load in N', color=colorLoad, 
                  va='center', ha='right', rotation=90)
# add grids, shaded area, ...
for ax in axTestCases:
    ax.grid()
    ax.axvspan(startTime, stopTime, color='gray', alpha=0.2, label=slidingStr, zorder=0)
    ax.legend(loc='upper left')
figTestCases.tight_layout(rect=[0, 0, 0.96, 1])
SaveFig(figTestCases, figPath, 'E_testCases', closeFig=False)

#%% do convergence analysis to get good amount of ANCF and beam elements
PrintHeadlineToConsole(text='Convergence Analysis')
# set load and slidingJoint function for convergence analysis
LoadTipFunctionConvergenceAnalysis = LoadTipFunctionEvolved
SlidingJointFunctionConvergenceAnalysis = SlidingJointFunctionEvolved
if not onlyLoadResults:
    # initialize beam (to inherit some data to ANCF beam) and initialize mbs
    beamConvergenceAnalysis = EulerBernulliBeam(beamTemplate=beamTemplateRubber)
    SC = exu.SystemContainer()
    mbs = SC.AddSystem()
    
    #++++++++++++++++++++++++++++++++++++++++++++++++++
    # ANCF elements
    nElementsANCFList = np.linspace(2, 40, 20, dtype=int) # hard coded!
    tipDisplacementConvergenceAnalysisANCFList = []
    for i, nElementsANCF in enumerate(nElementsANCFList):
        # retreive reference solution via ANCF beam with sliding joint
        mbs.Reset()
        nGround = mbs.AddNode(NodePointGround())
        # this is needed to constrain ANCF beam on the 'left' side
        mCoordGround = mbs.AddMarker(MarkerNodeCoordinate(nodeNumber=nGround, coordinate=0,
                                                          visualization=VMarkerNodeCoordinates(show=False)))
        sensorOutputVariableTypeConvergenceAnalysisANCFList = [exu.OutputVariableType.Displacement] # here, we only need displacement
        sDictConvergenceAnalysisANCF = AddANCFBeamWithSlidingJointToMBSAndAddSensors(mbs, SlidingJointFunctionConvergenceAnalysis, LoadTipFunctionConvergenceAnalysis, length=length, rhoA=beamConvergenceAnalysis.rhoA, EI=beamConvergenceAnalysis.EI, EA=beamConvergenceAnalysis.EA,
                                                                                     nElementsANCF=nElementsANCF, mCoordGround=mCoordGround, sensorOutputVariableTypeANCFList=sensorOutputVariableTypeConvergenceAnalysisANCFList,
                                                                                     beamNodeReferencePositionList=beamConvergenceAnalysis.nodeReferencePositionsList)
        mbs.Assemble()
        simulationSettings = GetStandardSimulationAndSetVisualizationSettings(tEnd, h, exportFrames=False)
        mbs.SolveDynamic(simulationSettings=simulationSettings)
        outputDictConvergenceAnalysisANCF = GetOutputDictANCFFromSDictANCF(mbs, sDictANCF=sDictConvergenceAnalysisANCF, sensorOutputVariableTypeANCFList=sensorOutputVariableTypeConvergenceAnalysisANCFList)
        tipDisplacementConvergenceAnalysisANCFList += [outputDictConvergenceAnalysisANCF['OutputVariableType.Displacement'][:, -2]]
        print('ANCF convergence analysis: Finished', i+1, '/', len(nElementsANCFList), 'analyses.')
    errorListConvergenceAnalysisANCF = []
    for tipDisplacementConvergenceAnalysisANCF in tipDisplacementConvergenceAnalysisANCFList[:-1]:
        errorListConvergenceAnalysisANCF += [AbsError(tipDisplacementConvergenceAnalysisANCF, tipDisplacementConvergenceAnalysisANCFList[-1])]
    errorListConvergenceAnalysisANCF += [0]
    AppendResultsToPKLFile(resultsFileName, 'convergenceAnalysis', {'nElementsANCFList': nElementsANCFList,
                                                                    'errorListConvergenceAnalysisANCF': errorListConvergenceAnalysisANCF})
    
    #++++++++++++++++++++++++++++++++++++++++++++++++++
    # beam elements
    nElementsBeamList = np.linspace(2, 60, 30, dtype=int) # hard coded!
    tipDisplacementConvergenceAnalysisBeamList = []
    for i, nElementsBeam in enumerate(nElementsBeamList):
        
        #++++++++++++++++++++++++++++++++++++++++++++++++++
        # initialize beam
        mbs.Reset()
        nGround = mbs.AddNode(NodePointGround())
        mCoordsGround = mbs.AddMarker(MarkerNodeCoordinates(nodeNumber=nGround, visualization=VMarkerNodeCoordinates(show=False)))
        beamConvergenceAnalysis = EulerBernulliBeam(beamTemplate=beamTemplateRubber,
                                                    nElements=nElementsBeam, # OVERWRITE!
                                                    slidingJointFunction=SlidingJointFunctionConvergenceAnalysis,
                                                    loadTipFunction=LoadTipFunctionConvergenceAnalysis,
                                                    interpolateBCForSlidingJoint=True,
                                                    fixBCWhenSliderNotMoving=False)
        
        #++++++++++++++++++++++++++++++++++++++++++++++++++
        # add beam to mbs
        sCoordinatesConvergenceAnalysisBeam, sCoordinates_tConvergenceAnalysisBeam = beamConvergenceAnalysis.AddBeamToMBSAndAddSensors(mbs, mCoordsGround=mCoordsGround, show=False)
        
        #++++++++++++++++++++++++++++++++++++++++++++++++++
        # simulate dynamically (WYSWYS)
        mbs.Assemble()
        simulationSettings = GetStandardSimulationAndSetVisualizationSettings(tEnd, h, exportFrames=False)
        mbs.SolveDynamic(simulationSettings=simulationSettings)
        
        #++++++++++++++++++++++++++++++++++++++++++++++++++
        # get results
        _, time, qConvergenceAnalysis, q_tConvergenceAnalysis = beamConvergenceAnalysis.GetTimeAndCoordinates(mbs, sCoordinatesConvergenceAnalysisBeam, sCoordinates_tConvergenceAnalysisBeam)
        tipDisplacementConvergenceAnalysisBeamList += [qConvergenceAnalysis[:, -2]]
        print('Beam convergence analysis: Finished', i+1, '/', len(nElementsBeamList), 'analyses.')
    errorListConvergenceAnalysisBeam = []
    for tipDisplacementConvergenceAnalysisBeam in tipDisplacementConvergenceAnalysisBeamList[:-1]:
        errorListConvergenceAnalysisBeam += [AbsError(tipDisplacementConvergenceAnalysisBeam, tipDisplacementConvergenceAnalysisBeamList[-1])]
    errorListConvergenceAnalysisBeam += [0]
    AppendResultsToPKLFile(resultsFileName, 'convergenceAnalysis', {'nElementsBeamList': nElementsBeamList,
                                                                    'errorListConvergenceAnalysisBeam': errorListConvergenceAnalysisBeam})
    
    #++++++++++++++++++++++++++++++++++++++++++++++++++
    # for debugging: plot displacement of tip
    if False:
        figDisplacementForDebugging, axDisplacementForDebugging = plt.subplots(1, 1, figsize=[textWidth, 3])
        for i in range(len(nElementsBeamList)):
            axDisplacementForDebugging.plot(time, tipDisplacementConvergenceAnalysisANCFList[i], label='ANCF ele: ' + str(nElementsANCFList[i]))
            axDisplacementForDebugging.plot(time, tipDisplacementConvergenceAnalysisBeamList[i], label='Beam ele: ' + str(nElementsBeamList[i]))
        axDisplacementForDebugging.legend()

#++++++++++++++++++++++++++++++++++++++++++++++++++
# plot the convergence analysis
resultsDictConvergenceAnalysis = GetResultsDictFromPKLFile(resultsFileName, 'convergenceAnalysis')
nElementsANCFList = resultsDictConvergenceAnalysis['nElementsANCFList']
nElementsBeamList = resultsDictConvergenceAnalysis['nElementsBeamList']
errorListConvergenceAnalysisANCF = resultsDictConvergenceAnalysis['errorListConvergenceAnalysisANCF']
errorListConvergenceAnalysisBeam = resultsDictConvergenceAnalysis['errorListConvergenceAnalysisBeam']
figConvergenceAnalysisANCF, axConvergenceAnalysisANCF = plt.subplots(1, 1, figsize=[textWidth, 3])
axConvergenceAnalysisANCF.plot(nElementsANCFList[:-1], errorListConvergenceAnalysisANCF[:-1], '-o', label=labelList[0])
axConvergenceAnalysisANCF.plot(nElementsBeamList[:-1], errorListConvergenceAnalysisBeam[:-1], '-*', label=labelList[1])
axConvergenceAnalysisANCF.set_yscale('log')
axConvergenceAnalysisANCF.grid()
axConvergenceAnalysisANCF.legend()
axConvergenceAnalysisANCF.set_xlabel('number of elements')
axConvergenceAnalysisANCF.set_ylabel('abs. error tip y-displacement in m')
SaveFig(figConvergenceAnalysisANCF, figPath, 'R_convergenceAnalysis', closeFig=False)

#%% test the importance of global modes by taking them away one after each other
PrintHeadlineToConsole(text='Global Mode Number Test')
# initial guess
nSnapshots = 6
nEigenmodesPerSnapshot = 6
nGlobalModes = nSnapshots * nEigenmodesPerSnapshot # max number
# initialize beam (to inherit some data to ANCF beam) and initialize mbs
beamGlobalModeImportance = EulerBernulliBeam(beamTemplate=beamTemplateRubber)
SC = exu.SystemContainer()
mbs = SC.AddSystem()

#++++++++++++++++++++++++++++++++++++++++++++++++++
# retreive reference solution via ANCF beam with sliding joint
nGround = mbs.AddNode(NodePointGround())
# this is needed to constrain ANCF beam on the 'left' side
mCoordGround = mbs.AddMarker(MarkerNodeCoordinate(nodeNumber=nGround, coordinate=0,
                                                  visualization=VMarkerNodeCoordinates(show=False)))
sensorOutputVariableTypeGlobalModeImportanceANCFList = [exu.OutputVariableType.Displacement] # here, we only need displacement
sDictGlobalModeImportanceANCF = AddANCFBeamWithSlidingJointToMBSAndAddSensors(mbs, SlidingJointFunctionEvolved, LoadTipFunctionEvolved, length=length, rhoA=beamGlobalModeImportance.rhoA, EI=beamGlobalModeImportance.EI, EA=beamGlobalModeImportance.EA,
                                                                              nElementsANCF=nElementsANCF, mCoordGround=mCoordGround, sensorOutputVariableTypeANCFList=sensorOutputVariableTypeGlobalModeImportanceANCFList,
                                                                              beamNodeReferencePositionList=beamGlobalModeImportance.nodeReferencePositionsList)
mbs.Assemble()
simulationSettings = GetStandardSimulationAndSetVisualizationSettings(tEnd, h, exportFrames=False)
mbs.SolveDynamic(simulationSettings=simulationSettings)
outputDictGlobalModeImportanceANCF = GetOutputDictANCFFromSDictANCF(mbs, sDictANCF=sDictGlobalModeImportanceANCF, sensorOutputVariableTypeANCFList=sensorOutputVariableTypeGlobalModeImportanceANCFList)
displacementGlobalModeImportanceANCF = outputDictGlobalModeImportanceANCF['OutputVariableType.Displacement']

#++++++++++++++++++++++++++++++++++++++++++++++++++
# get snapshots of beam (offline to save time) - we cannot use the wrapper since the wrapper already reduces
eigenmodesLiftedAndMassNormalizedList = beamGlobalModeImportance.GetGlobalMassNormalizedEigenmodesListForNumberOfConstrainedNodes(nNodesConstrained=nSnapshots,
                                                                                                                                                             nEigenmodes=nEigenmodesPerSnapshot,
                                                                                                                                                             verbose=True)
globalReductionBasis = beamGlobalModeImportance.GetGlobalReductionBasisUsingEigenmodesList(eigenmodesList=eigenmodesLiftedAndMassNormalizedList,
                                                                                              nGlobalModes=nGlobalModes, verbose=True)

#++++++++++++++++++++++++++++++++++++++++++++++++++
# iterate over the global modes and remove them successively
errorList = []
modesUsedList = []
for i in range(nGlobalModes-2):
    #++++++++++++++++++++++++++++++++++++++++++++++++++
    # initialize beam
    mbs.Reset()
    nGround = mbs.AddNode(NodePointGround())
    mCoordsGround = mbs.AddMarker(MarkerNodeCoordinates(nodeNumber=nGround, visualization=VMarkerNodeCoordinates(show=False)))
    beamGlobalModeImportance = EulerBernulliBeam(beamTemplate=beamTemplateRubber,
                                                    slidingJointFunction=SlidingJointFunctionEvolved,
                                                    loadTipFunction=LoadTipFunctionEvolved,
                                                    interpolateBCForSlidingJoint=True,
                                                    fixBCWhenSliderNotMoving=False)
    
    #++++++++++++++++++++++++++++++++++++++++++++++++++
    # GLOBALLY reduced beam and add it
    nGlobalModesUsed = nGlobalModes - i
    modesUsedList += [nGlobalModesUsed]
    beamGlobalModeImportance.ReduceBeam(reductionBasis=globalReductionBasis[:, 0:nGlobalModesUsed])
    sCoordinatesGlobalModeImportancePMR, sCoordinates_tGlobalModeImportancePMR = beamGlobalModeImportance.AddBeamToMBSAndAddSensors(mbs, mCoordsGround=mCoordsGround, show=False)
    
    #++++++++++++++++++++++++++++++++++++++++++++++++++
    # simulate dynamically (WYSWYS)
    mbs.Assemble()
    simulationSettings = GetStandardSimulationAndSetVisualizationSettings(tEnd, h, exportFrames=False)
    mbs.SolveDynamic(simulationSettings=simulationSettings)
    
    #++++++++++++++++++++++++++++++++++++++++++++++++++
    # get results
    _, time, qGlobalModeImportancePMR, q_tGlobalModeImportancePMR = beamGlobalModeImportance.GetTimeAndCoordinates(mbs, sCoordinatesGlobalModeImportancePMR, sCoordinates_tGlobalModeImportancePMR)
    errorList += [AbsError(qGlobalModeImportancePMR[:, 1::beamGlobalModeImportance.nDOFsPerNode], displacementGlobalModeImportanceANCF[:, 1::3])]
    print('Global mode number test: Finished', i+1, '/', nGlobalModes-2, 'analyses.')

#++++++++++++++++++++++++++++++++++++++++++++++++++
# plot the errors
figGlobalModesImportance, axGlobalModesImportance = plt.subplots(1, 1, figsize=[textWidth, 3])
x = np.arange(len(modesUsedList))
axGlobalModesImportance.bar(x, errorList)
axGlobalModesImportance.set_xticks(x, modesUsedList)
axGlobalModesImportance.set_xlabel('number of used modes')
axGlobalModesImportance.set_ylabel('abs. error y-displacement in m')
axGlobalModesImportance.grid()

#%% 
# # using the estimated amount of global modes, we try combinations
# nGlobalModes = 
# nSnapshotsList = []
# nEigenmodesPerSnapshotList = []

# #%% test the importance of global modes by successively removing one at a time
# # initialize beam (to inherit some data to ANCF beam) and mbs
# beamModeImportancePMR = EulerBernulliBeam(beamTemplate=beamTemplateRubber,
#                                           slidingJointFunction=SlidingJointTestFunction, loadTipFunction=LoadTipTestFunction,
#                                           interpolateBCForSlidingJoint=True,
#                                           fixBCWhenSliderNotMoving=False)
# SC = exu.SystemContainer()
# mbs = SC.AddSystem()

# #++++++++++++++++++++++++++++++++++++++++++++++++++
# # retreive reference solution via ANCF beam with sliding joint
# nGround = mbs.AddNode(NodePointGround())
# # this is needed to constrain ANCF beam on the 'left' side
# mCoordGround = mbs.AddMarker(MarkerNodeCoordinate(nodeNumber=nGround, coordinate=0,
#                                                   visualization=VMarkerNodeCoordinates(show=False)))
# sensorOutputVariableTypeVariationANCFList = [exu.OutputVariableType.Displacement] # here, we only need displacement
# sDictVariationANCF = AddANCFBeamWithSlidingJointToMBSAndAddSensors(mbs, SlidingJointTestFunction, LoadTipTestFunction, length=length, rhoA=beamModeImportancePMR.rhoA, EI=beamModeImportancePMR.EI, EA=beamModeImportancePMR.EA,
#                                                                    nElementsANCF=10, mCoordGround=mCoordGround, sensorOutputVariableTypeANCFList=sensorOutputVariableTypeVariationANCFList,
#                                                                    beamNodeReferencePositionList=beamModeImportancePMR.nodeReferencePositionsList)
# mbs.Assemble()
# simulationSettings = GetStandardSimulationAndSetVisualizationSettings(tEnd, h, exportFrames=False)
# mbs.SolveDynamic(simulationSettings=simulationSettings)
# outputDictVariationANCF = GetOutputDictANCFFromSDictANCF(mbs, sDictANCF=sDictVariationANCF, sensorOutputVariableTypeANCFList=sensorOutputVariableTypeVariationANCFList)
# displacementImportanceANCF = outputDictVariationANCF['OutputVariableType.Displacement']

# #++++++++++++++++++++++++++++++++++++++++++++++++++
# # get snapshots of beam (offline to save time)
# nEigenmodesPerSnapshot = 4
# nSnapshots = 6
# nGlobalModes = 24
# eigenmodesLiftedAndMassNormalizedListPMR = beamModeImportancePMR.GetGlobalMassNormalizedEigenmodesListForNumberOfConstrainedNodes(nNodesConstrained=nSnapshots,
#                                                                                                                                   nEigenmodes=nEigenmodesPerSnapshot,
#                                                                                                                                   verbose=True)
# globalReductionBasisImportance = beamModeImportancePMR.GetGlobalReductionBasisUsingEigenmodesList(eigenmodesList=eigenmodesLiftedAndMassNormalizedListPMR,
#                                                                                                   nGlobalModes=nGlobalModes, verbose=True)

# #++++++++++++++++++++++++++++++++++++++++++++++++++
# # iterate over the global modes and remove them successively (always leave one mode)
# errorList = []
# modesUsedList = []
# for i in range(nGlobalModes):
#     print(i)
#     mbs.Reset()
#     nGround = mbs.AddNode(NodePointGround())
#     mCoordsGround = mbs.AddMarker(MarkerNodeCoordinates(nodeNumber=nGround, visualization=VMarkerNodeCoordinates(show=False)))
#     beamModeImportancePMR = EulerBernulliBeam(beamTemplate=beamTemplateRubber,
#                                               slidingJointFunction=SlidingJointTestFunction, loadTipFunction=LoadTipTestFunction,
#                                               interpolateBCForSlidingJoint=True,
#                                               fixBCWhenSliderNotMoving=False)
    
#     #++++++++++++++++++++++++++++++++++++++++++++++++++
#     # GLOBALLY reduced beam and add it
#     beamModeImportancePMR.ReduceBeam(reductionBasis=np.delete(globalReductionBasisImportance, i, axis=1))
#     sCoordinatesImportancePMR, sCoordinates_tImportancePMR = beamModeImportancePMR.AddBeamToMBSAndAddSensors(mbs, mCoordsGround=mCoordsGround, show=False)
    
#     #++++++++++++++++++++++++++++++++++++++++++++++++++
#     # simulate dynamically (WYSWYS)
#     mbs.Assemble()
#     simulationSettings = GetStandardSimulationAndSetVisualizationSettings(tEnd, h, exportFrames=False)
#     mbs.SolveDynamic(simulationSettings=simulationSettings)
    
#     #++++++++++++++++++++++++++++++++++++++++++++++++++
#     # get results and compute Normalized Root Mean Square Error (NRMSE)
#     _, time, qImportancePMR, q_tImportancePMR = beamModeImportancePMR.GetTimeAndCoordinates(mbs, sCoordinatesImportancePMR, sCoordinates_tImportancePMR)
#     errorList += [AbsError(qImportancePMR[:, 1::beamModeImportancePMR.nDOFsPerNode], displacementImportanceANCF[:, 1::3])]

# fig, ax = plt.subplots(1, 1, figsize=[6, 3])
# x = np.arange(1, nGlobalModes+1)
# ax.bar(x, errorList)
# # ax.set_xticks(x, modesUsedList)
# # ax.set_yscale('log')
# ax.set_xlabel('deleted mode')
# ax.set_ylabel('absoulte error tip displacement in m')
# ax.grid()

# #%% variation experiment to test good number of snapshots, eigenmodes per snapshot, and number of global modes
# PrintHeadlineToConsole(text='Variation Experiment')
# # set up the parameters for the experiment
# loadResults = True
# nSnapshotsList = [4, 8, 12, 16]
# nEigenmodesPerSnapshotList = [3, 4, 5, 6]
# nGlobalModesList = [4, 8, 12, 16, 20, 24, 28, 32]
# if not loadResults:
#     # precompute info for status bar
#     nValidVariations = sum(1 for nSnapshots in nSnapshotsList for nEigenmodesPerSnapshot in nEigenmodesPerSnapshotList for nGlobalModes in nGlobalModesList
#                            if nGlobalModes <= (nEigenmodesPerSnapshot * nSnapshots))
    
#     #++++++++++++++++++++++++++++++++++++++++++++++++++
#     # initialize beam (to inherit some data to ANCF beam) and mbs
#     beamVariationPMR = EulerBernulliBeam(beamTemplate=beamTemplateRubber,
#                                          slidingJointFunction=SlidingJointTestFunction, loadTipFunction=LoadTipTestFunction,
#                                          interpolateBCForSlidingJoint=True,
#                                          fixBCWhenSliderNotMoving=False)
#     SC = exu.SystemContainer()
#     mbs = SC.AddSystem()
    
#     #++++++++++++++++++++++++++++++++++++++++++++++++++
#     # retreive reference solution via ANCF beam with sliding joint
#     nGround = mbs.AddNode(NodePointGround())
#     # this is needed to constrain ANCF beam on the 'left' side
#     mCoordGround = mbs.AddMarker(MarkerNodeCoordinate(nodeNumber=nGround, coordinate=0,
#                                                       visualization=VMarkerNodeCoordinates(show=False)))
#     sensorOutputVariableTypeVariationANCFList = [exu.OutputVariableType.Displacement] # here, we only need displacement
#     sDictVariationANCF = AddANCFBeamWithSlidingJointToMBSAndAddSensors(mbs, SlidingJointTestFunction, LoadTipTestFunction, length=length, rhoA=beamVariationPMR.rhoA, EI=beamVariationPMR.EI, EA=beamVariationPMR.EA,
#                                                                        nElementsANCF=10, mCoordGround=mCoordGround, sensorOutputVariableTypeANCFList=sensorOutputVariableTypeVariationANCFList,
#                                                                        beamNodeReferencePositionList=beamVariationPMR.nodeReferencePositionsList)
#     mbs.Assemble()
#     simulationSettings = GetStandardSimulationAndSetVisualizationSettings(tEnd, h, exportFrames=False)
#     mbs.SolveDynamic(simulationSettings=simulationSettings)
#     outputDictVariationANCF = GetOutputDictANCFFromSDictANCF(mbs, sDictANCF=sDictVariationANCF, sensorOutputVariableTypeANCFList=sensorOutputVariableTypeVariationANCFList)
#     displacementVariationANCF = outputDictVariationANCF['OutputVariableType.Displacement']
    
#     #++++++++++++++++++++++++++++++++++++++++++++++++++
#     # do variation
#     variationResults = np.full((len(nSnapshotsList), len(nEigenmodesPerSnapshotList), len(nGlobalModesList)), -1.0)
#     with tqdm(total=nValidVariations, desc='Parameter variation', unit='case') as pbar:
#         for i, nSnapshots in enumerate(nSnapshotsList):
#             for j, nEigenmodesPerSnapshot in enumerate(nEigenmodesPerSnapshotList):
#                 for k, nGlobalModes in enumerate(nGlobalModesList):
#                     if nGlobalModes <= (nEigenmodesPerSnapshot * nSnapshots):
                        
#                         #++++++++++++++++++++++++++++++++++++++++++++++++++
#                         # set up mbs and beam for this variation
#                         mbs.Reset()
#                         nGround = mbs.AddNode(NodePointGround())
#                         mCoordsGround = mbs.AddMarker(MarkerNodeCoordinates(nodeNumber=nGround, visualization=VMarkerNodeCoordinates(show=False)))
#                         beamVariationPMR = EulerBernulliBeam(beamTemplate=beamTemplateRubber,
#                                                              slidingJointFunction=SlidingJointTestFunction, loadTipFunction=LoadTipTestFunction,
#                                                              interpolateBCForSlidingJoint=True,
#                                                              fixBCWhenSliderNotMoving=False)
                        
#                         #++++++++++++++++++++++++++++++++++++++++++++++++++
#                         # GLOBALLY reduced beam
#                         beamVariationPMR.GlobalReduceBeamWrapper(nGlobalModes=nGlobalModes, nSnapshots=nSnapshots, nEigenmodesPerSnapshot=nEigenmodesPerSnapshot)
#                         sCoordinatesVariationPMR, sCoordinates_tVariationPMR = beamVariationPMR.AddBeamToMBSAndAddSensors(mbs, mCoordsGround=mCoordsGround, show=False)
                        
#                         #++++++++++++++++++++++++++++++++++++++++++++++++++
#                         # simulate dynamically (WYSWYS)
#                         mbs.Assemble()
#                         simulationSettings = GetStandardSimulationAndSetVisualizationSettings(tEnd, h, exportFrames=False)
#                         mbs.SolveDynamic(simulationSettings=simulationSettings)
                        
#                         #++++++++++++++++++++++++++++++++++++++++++++++++++
#                         # get results and compute Normalized Root Mean Square Error (NRMSE)
#                         _, time, qVariationPMR, q_tVariationPMR = beamVariationPMR.GetTimeAndCoordinates(mbs, sCoordinatesVariationPMR, sCoordinates_tVariationPMR)
#                         variationResults[i, j, k] = NRMSEVector(qVariationPMR[:, 1::beamVariationPMR.nDOFsPerNode], displacementVariationANCF[:, 1::3])
                        
#                         #++++++++++++++++++++++++++++++++++++++++++++++++++
#                         # update pbar
#                         pbar.set_postfix({'nSnap': nSnapshots, 'nEig': nEigenmodesPerSnapshot, 'nGlob': nGlobalModes})
#                         pbar.update(1)
#     np.save('variationSolution', variationResults)
    
# #++++++++++++++++++++++++++++++++++++++++++++++++++
# # do plotting
# variationResults = np.load('variationSolution.npy')
# validResults = variationResults[variationResults != -1.0]
# vmin, vmax = validResults.min(), validResults.max()
# cmap = cm.RdYlGn_r
# norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
# figVariation = plt.figure(figsize=[textWidth, 6])
# axVariation = figVariation.add_subplot(111, projection='3d')
# for i, nSnap in enumerate(nSnapshotsList):
#     for j, nEig in enumerate(nEigenmodesPerSnapshotList):
#         for k, nGlob in enumerate(nGlobalModesList):
#             val = variationResults[i, j, k]
#             if val == -1.0:
#                 continue
#             axVariation.scatter(nSnap, nEig, nGlob, c=[cmap(norm(val))], s=60, alpha=0.8)
# axVariation.set_xlabel('num. snapshots')
# axVariation.set_ylabel('num. eigenmodes / snapshot')
# axVariation.set_zlabel('num. global modes')
# sm = cm.ScalarMappable(cmap=cmap, norm=norm)
# cb = figVariation.colorbar(sm, ax=axVariation, orientation='horizontal', label='normalized RMSE of beam displ.', pad=0.10, shrink=0.6)
# ticks = np.linspace(vmin, vmax, 4)
# cb.set_ticks(ticks)
# cb.ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
# plt.tight_layout
# plt.show()
# SaveFig(figVariation, figPath, 'R_variation', closeFig=True)



# #%% pick one combination and make plot of tip displacement
# # nSnapshots = 12             # number of snapshots used for global reduction
# # nEigenmodesPerSnapshot = 4  # eigenmodes used per snapshot
# # nGlobalModes = 24           # number of global modes in global reduction basis

# # #++++++++++++++++++++++++++++++++++++++++++++++++++
# # # initialize beam and mbs
# # beamVariationPMR = EulerBernulliBeam(beamTemplate=beamTemplateRubber,
# #                                      slidingJointFunction=SlidingJointTestFunction, loadTipFunction=LoadTipTestFunction,
# #                                      interpolateBCForSlidingJoint=True)
# # SC = exu.SystemContainer()
# # mbs = SC.AddSystem()













# #%%

# nCases = 3
# computationTimeListANCF = []
# computationTimeListNR = []
# computationTimeListPMR = []
# nSnapshots = 12
# nEigenmodesPerSnapshot = 4
# nGlobalModes = 24
# SC = exu.SystemContainer()
# mbs = SC.AddSystem()
# beamComputationNR = EulerBernulliBeam(beamTemplate=beamTemplateRubber,
#                                       slidingJointFunction=SlidingJointTestFunction, loadTipFunction=LoadTipTestFunction,
#                                       interpolateBCForSlidingJoint=True,
#                                       fixBCWhenSliderNotMoving=False)
# for i in range(nCases):
#     mbs.Reset()
#     nGround = mbs.AddNode(NodePointGround())
#     mCoordGround = mbs.AddMarker(MarkerNodeCoordinate(nodeNumber=nGround, coordinate=0,
#                                                       visualization=VMarkerNodeCoordinates(show=False)))
#     sensorOutputVariableTypeVariationANCFList = [exu.OutputVariableType.Displacement] # here, we only need displacement
#     sDictVariationANCF = AddANCFBeamWithSlidingJointToMBSAndAddSensors(mbs, SlidingJointTestFunction, LoadTipTestFunction, length=length, rhoA=beamComputationNR.rhoA, EI=beamComputationNR.EI, EA=beamComputationNR.EA,
#                                                                        nElementsANCF=10, mCoordGround=mCoordGround, sensorOutputVariableTypeANCFList=sensorOutputVariableTypeVariationANCFList,
#                                                                        beamNodeReferencePositionList=beamComputationNR.nodeReferencePositionsList)
#     mbs.Assemble()
#     simulationSettings = GetStandardSimulationAndSetVisualizationSettings(tEnd, h, exportFrames=False)
#     startTime = tm.time()
#     mbs.SolveDynamic(simulationSettings=simulationSettings)
#     duration = tm.time() - startTime
#     computationTimeListANCF += [duration]
# for i in range(nCases):
#     mbs.Reset()
#     nGround = mbs.AddNode(NodePointGround())
#     mCoordsGround = mbs.AddMarker(MarkerNodeCoordinates(nodeNumber=nGround, visualization=VMarkerNodeCoordinates(show=False)))
#     beamComputationNR = EulerBernulliBeam(beamTemplate=beamTemplateRubber,
#                                           slidingJointFunction=SlidingJointTestFunction, loadTipFunction=LoadTipTestFunction,
#                                           interpolateBCForSlidingJoint=True,
#                                           fixBCWhenSliderNotMoving=False)
#     _, _ = beamComputationNR.AddBeamToMBSAndAddSensors(mbs, mCoordsGround=mCoordsGround, show=False)
    
#     #++++++++++++++++++++++++++++++++++++++++++++++++++
#     # simulate dynamically (WYSWYS)
#     mbs.Assemble()
#     simulationSettings = GetStandardSimulationAndSetVisualizationSettings(tEnd, h, exportFrames=False)
#     # supress warnings here
#     startTime = tm.time()
#     mbs.SolveDynamic(simulationSettings=simulationSettings)
#     duration = tm.time() - startTime
#     computationTimeListNR += [duration]
# for i in range(nCases):
#     mbs.Reset()
#     nGround = mbs.AddNode(NodePointGround())
#     mCoordsGround = mbs.AddMarker(MarkerNodeCoordinates(nodeNumber=nGround, visualization=VMarkerNodeCoordinates(show=False)))
#     beamComputationPMR = EulerBernulliBeam(beamTemplate=beamTemplateRubber,
#                                            slidingJointFunction=SlidingJointTestFunction, loadTipFunction=LoadTipTestFunction,
#                                            interpolateBCForSlidingJoint=True,
#                                            fixBCWhenSliderNotMoving=False)
    
#     #++++++++++++++++++++++++++++++++++++++++++++++++++
#     # GLOBALLY reduced beam
#     beamComputationPMR.GlobalReduceBeamWrapper(nGlobalModes=nGlobalModes, nSnapshots=nSnapshots, nEigenmodesPerSnapshot=nEigenmodesPerSnapshot)
#     _, _ = beamComputationPMR.AddBeamToMBSAndAddSensors(mbs, mCoordsGround=mCoordsGround, show=False)
    
#     #++++++++++++++++++++++++++++++++++++++++++++++++++
#     # simulate dynamically (WYSWYS)
#     mbs.Assemble()
#     simulationSettings = GetStandardSimulationAndSetVisualizationSettings(tEnd, h, exportFrames=False)
#     startTime = tm.time()
#     mbs.SolveDynamic(simulationSettings=simulationSettings)
#     duration = tm.time() - startTime
#     computationTimeListPMR += [duration]

# #%%
# # Calculate means and standard deviations
# means = [np.mean(computationTimeListANCF), np.mean(computationTimeListNR), np.mean(computationTimeListPMR)]
# # stds = [np.mean(computationTimeListANCF), np.std(computationTimeListNR), np.std(computationTimeListPMR)]
# # Bar positions
# x_pos = np.arange(len(means))
# # Create bar chart with error bars
# plt.bar(x_pos, means, capsize=10, alpha=0.7, color=[standardColors[0], standardColors[1], standardColors[2]])
# plt.xticks(x_pos, ['ANCF', 'non-reduced', 'globally reduced'])
# plt.ylabel('mean computation time in s')
# plt.show()
# plt.grid()