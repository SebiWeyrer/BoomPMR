#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#
# Details:  
# Author:   Sebastian Weyrer
# Date:     2026-04-27
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
onlyLoadResults = True # if False, expect long execution time due to expensive tests

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
    beamConvergenceAnalysis = EulerBernulliBeam(beamTemplate=beamTemplateRubber)
    
    #++++++++++++++++++++++++++++++++++++++++++++++++++
    # ANCF elements
    nElementsANCFList = np.linspace(2, 44, 22, dtype=int) # hard coded!
    tipDisplacementConvergenceAnalysisANCFList = []
    for i, nElementsANCF in enumerate(nElementsANCFList):
        # retreive reference solution via ANCF beam with sliding joint
        SC = exu.SystemContainer()
        mbs = SC.AddSystem() # always create new mbs (or mbs.Reset())
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
        # initialize beam
        SC = exu.SystemContainer()
        mbs = SC.AddSystem() # always create new mbs (or mbs.Reset())
        nGround = mbs.AddNode(NodePointGround())
        mCoordsGround = mbs.AddMarker(MarkerNodeCoordinates(nodeNumber=nGround, visualization=VMarkerNodeCoordinates(show=False)))
        beamConvergenceAnalysis = EulerBernulliBeam(beamTemplate=beamTemplateRubber,
                                                    nElements=nElementsBeam, # OVERWRITE!
                                                    slidingJointFunction=SlidingJointFunctionConvergenceAnalysis,
                                                    loadTipFunction=LoadTipFunctionConvergenceAnalysis)
        
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
figConvergenceAnalysisANCF, axConvergenceAnalysisANCF = plt.subplots(1, 1, figsize=[textWidth, 2.5])
axConvergenceAnalysisANCF.loglog(nElementsANCFList[:-1], errorListConvergenceAnalysisANCF[:-1], '-o', label=labelList[0], color=standardColors[0])
axConvergenceAnalysisANCF.loglog(nElementsBeamList[:-1], errorListConvergenceAnalysisBeam[:-1], '-*', label=labelList[1], color=standardColors[1])
axConvergenceAnalysisANCF.grid()
axConvergenceAnalysisANCF.legend()
axConvergenceAnalysisANCF.set_xlabel('number of elements')
axConvergenceAnalysisANCF.set_ylabel('abs. error tip y-displacement in m')
SaveFig(figConvergenceAnalysisANCF, figPath, 'E_convergenceAnalysis', closeFig=False)

#%% test needed number of global modes by taking them away
PrintHeadlineToConsole(text='Global Mode Number')
if not onlyLoadResults:
    # initial guess
    nSnapshots = 6
    nEigenmodesPerSnapshot = 6
    nGlobalModesInitial = nSnapshots * nEigenmodesPerSnapshot # max number
    
    #++++++++++++++++++++++++++++++++++++++++++++++++++
    # offline compute initial global reduction basis
    beamGlobalModeNumber = EulerBernulliBeam(beamTemplate=beamTemplateRubber)
    eigenmodesLiftedAndMassNormalizedList = beamGlobalModeNumber.GetGlobalMassNormalizedEigenmodesListForNumberOfConstrainedNodes(nNodesConstrained=nSnapshots,
                                                                                                                                                                 nEigenmodes=nEigenmodesPerSnapshot,
                                                                                                                                                                 verbose=True)
    globalReductionBasisInitial = beamGlobalModeNumber.GetGlobalReductionBasisUsingEigenmodesList(eigenmodesList=eigenmodesLiftedAndMassNormalizedList,
                                                                                                      nGlobalModes=nGlobalModesInitial, verbose=True)
    
    #++++++++++++++++++++++++++++++++++++++++++++++++++
    # iterate over the global modes and remove them successively
    strainEnergySumList = []
    globalModesUsedList = []
    for i in range(nGlobalModesInitial-2): # leave at least two modes
        # initialize beam
        SC = exu.SystemContainer()
        mbs = SC.AddSystem() # always create new mbs (or mbs.Reset())
        nGround = mbs.AddNode(NodePointGround())
        mCoordsGround = mbs.AddMarker(MarkerNodeCoordinates(nodeNumber=nGround, visualization=VMarkerNodeCoordinates(show=False)))
        beamGlobalModeNumber = EulerBernulliBeam(beamTemplate=beamTemplateRubber,
                                                     slidingJointFunction=SlidingJointFunctionEvolved,
                                                     loadTipFunction=LoadTipFunctionEvolved)
        
        #++++++++++++++++++++++++++++++++++++++++++++++++++
        # GLOBALLY reduced beam and add it
        nGlobalModesUsed = nGlobalModesInitial - i
        globalModesUsedList += [nGlobalModesUsed]
        beamGlobalModeNumber.ReduceBeam(reductionBasis=globalReductionBasisInitial[:, 0:nGlobalModesUsed])
        sCoordinatesGlobalModeNumberBeamPMR, sCoordinates_tGlobalModeNumberBeamPMR = beamGlobalModeNumber.AddBeamToMBSAndAddSensors(mbs, mCoordsGround=mCoordsGround, show=False)
        
        #++++++++++++++++++++++++++++++++++++++++++++++++++
        # simulate dynamically (WYSWYS)
        mbs.Assemble()
        simulationSettings = GetStandardSimulationAndSetVisualizationSettings(tEnd, h, exportFrames=False)
        mbs.SolveDynamic(simulationSettings=simulationSettings)
        
        #++++++++++++++++++++++++++++++++++++++++++++++++++
        # get results
        _, time, qGlobalModeNumberBeamPMR, q_tGlobalModeNumberBeamPMR = beamGlobalModeNumber.GetTimeAndCoordinates(mbs, sCoordinatesGlobalModeNumberBeamPMR, sCoordinates_tGlobalModeNumberBeamPMR)
        strainEnergySum = beamGlobalModeNumber.GetSumOfBeamStrainEnergy(qGlobalModeNumberBeamPMR)
        strainEnergySumList += [strainEnergySum]
        print('Global mode number test: Finished', i+1, '/', nGlobalModesInitial-2, 'analyses.')
    strainEnergySumErrorList = []
    strainEnergySumErrorList += [0]
    for strainEnergySum in strainEnergySumList[1:]:
        strainEnergySumErrorList += [abs(strainEnergySum-strainEnergySumList[0])]
    AppendResultsToPKLFile(resultsFileName, 'globalModeNumber', {'globalModesUsedList': globalModesUsedList,
                                                                 'strainEnergySumErrorList': strainEnergySumErrorList})

#++++++++++++++++++++++++++++++++++++++++++++++++++
# plot the errors
indexOfMaxNumberOfGlobalModesPlotted = 14
resultsDictGlobalModeImportance = GetResultsDictFromPKLFile(resultsFileName, 'globalModeNumber')
globalModesUsedList = resultsDictGlobalModeImportance['globalModesUsedList']
strainEnergySumErrorList = resultsDictGlobalModeImportance['strainEnergySumErrorList']
figGlobalModesNumber, axGlobalModesNumber = plt.subplots(1, 1, figsize=[textWidth, 2])
x = np.arange(len(globalModesUsedList)-indexOfMaxNumberOfGlobalModesPlotted)
axGlobalModesNumber.bar(x, [strainEnergySumErrorList[i]/1e3 for i in range(len(strainEnergySumErrorList))][indexOfMaxNumberOfGlobalModesPlotted:],
                        color=standardColors[2])
axGlobalModesNumber.set_xticks(x, globalModesUsedList[indexOfMaxNumberOfGlobalModesPlotted:])
axGlobalModesNumber.set_xlabel(r'number $r_\mathrm{glob.}$ of used global modes')
axGlobalModesNumber.set_ylabel('error of strain energy in kJ')
axGlobalModesNumber.grid(axis='y')
axGlobalModesNumber.set_axisbelow(True)
SaveFig(figGlobalModesNumber, figPath, 'E_globalModesNumber', closeFig=False)

#%% test different combinations of nSnapshots and nEigenmodesPerSnapshot
PrintHeadlineToConsole(text='Combination Analysis: Number of and Modes per Snapshot')
