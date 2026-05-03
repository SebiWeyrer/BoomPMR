#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#
# Details:  ANCF ... ANCF beam; beam ... Euler-Bernoulli beam (NR or PMR)
# Author:   Sebastian Weyrer
# Date:     2026-05-03
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
    beamConvergenceAnalysisNR = EulerBernulliBeam(beamTemplate=beamTemplateRubber)
    
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
        sDictConvergenceAnalysisANCF = AddANCFBeamWithSlidingJointToMBSAndAddSensors(mbs, SlidingJointFunctionConvergenceAnalysis, LoadTipFunctionConvergenceAnalysis, length=length, rhoA=beamConvergenceAnalysisNR.rhoA, EI=beamConvergenceAnalysisNR.EI, EA=beamConvergenceAnalysisNR.EA,
                                                                                     nElementsANCF=nElementsANCF, mCoordGround=mCoordGround, sensorOutputVariableTypeANCFList=sensorOutputVariableTypeConvergenceAnalysisANCFList,
                                                                                     beamNodeReferencePositionList=beamConvergenceAnalysisNR.nodeReferencePositionsList)
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
        beamConvergenceAnalysisNR = EulerBernulliBeam(beamTemplate=beamTemplateRubber,
                                                      nElements=nElementsBeam, # OVERWRITE!
                                                      slidingJointFunction=SlidingJointFunctionConvergenceAnalysis,
                                                      loadTipFunction=LoadTipFunctionConvergenceAnalysis)
        
        #++++++++++++++++++++++++++++++++++++++++++++++++++
        # add beam to mbs
        sCoordinatesConvergenceAnalysisBeam, sCoordinates_tConvergenceAnalysisBeam = beamConvergenceAnalysisNR.AddBeamToMBSAndAddSensors(mbs, mCoordsGround=mCoordsGround, show=False)
        
        #++++++++++++++++++++++++++++++++++++++++++++++++++
        # simulate dynamically (WYSWYS)
        mbs.Assemble()
        simulationSettings = GetStandardSimulationAndSetVisualizationSettings(tEnd, h, exportFrames=False)
        mbs.SolveDynamic(simulationSettings=simulationSettings)
        
        #++++++++++++++++++++++++++++++++++++++++++++++++++
        # get results
        _, time, qConvergenceAnalysis, q_tConvergenceAnalysis = beamConvergenceAnalysisNR.GetTimeAndCoordinates(mbs, sCoordinatesConvergenceAnalysisBeam, sCoordinates_tConvergenceAnalysisBeam)
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

#%% compute strain eneregy sum error over number of global modes
PrintHeadlineToConsole(text='Strain Energy Sum Error')
# offline compute reference strain energy sum (non-reduced beam)
SC = exu.SystemContainer()
mbs = SC.AddSystem() # always create new mbs (or mbs.Reset())
nGround = mbs.AddNode(NodePointGround())
mCoordsGround = mbs.AddMarker(MarkerNodeCoordinates(nodeNumber=nGround, visualization=VMarkerNodeCoordinates(show=False)))
beamStrainEnergySumNR = EulerBernulliBeam(beamTemplate=beamTemplateRubber,
                                          slidingJointFunction=SlidingJointFunctionEvolved,
                                          loadTipFunction=LoadTipFunctionEvolved)
sCoordinatesStrainEnergySumBeamNR, sCoordinates_tStrainEnergySumBeamNR = beamGlobalModeNumberNRNR.AddBeamToMBSAndAddSensors(mbs, mCoordsGround=mCoordsGround, show=False)
mbs.Assemble()
simulationSettings = GetStandardSimulationAndSetVisualizationSettings(tEnd, h, exportFrames=False)
mbs.SolveDynamic(simulationSettings=simulationSettings)
_, time, qStrainEnergySumBeamNR, q_tStrainEnergySumBeamNR = beamStrainEnergySumNR.GetTimeAndCoordinates(mbs, sCoordinatesStrainEnergySumBeamNR, sCoordinates_tStrainEnergySumBeamNR)
strainEnergySumNR = beamStrainEnergySumNR.GetSumOfBeamStrainEnergy(qStrainEnergySumBeamNR)
print('Non-reduced strain energy sum:', strainEnergySumNR, 'J.')

#++++++++++++++++++++++++++++++++++++++++++++++++++
# do MAIN variation for nSnap and rSnap
# pairs that should be investigated (first number is nSnap, second is rSnap)
pairsToBeInvestigatedList = [(6, 6), (4, 9), (9, 4), (2, 18), (18, 2)]
nGlobalModesInitial = 36 # here we always use combinations that give 36 (could be any number...)
nGlobalModesUsedList = [nGlobalModesInitial-i for i in range(nGlobalModesInitial-2)]
nAnalyses = len(nGlobalModesUsedList) * len(pairsToBeInvestigatedList)
strainEnergySumList = []
nAnalysesCompleted = 0
for pair in pairsToBeInvestigatedList:
    # offline compute initial global reduction basis
    beamStrainEnergyPMR = EulerBernulliBeam(beamTemplate=beamTemplateRubber)
    eigenmodesLiftedAndMassNormalizedList = beamStrainEnergyPMR.GetGlobalMassNormalizedEigenmodesListForNumberOfConstrainedNodes(nNodesConstrained=pair[0],
                                                                                                                                 nEigenmodes=pair[1],
                                                                                                                                 verbose=True)
    globalReductionBasisInitial = beamStrainEnergyPMR.GetGlobalReductionBasisUsingEigenmodesList(eigenmodesList=eigenmodesLiftedAndMassNormalizedList,
                                                                                                 nGlobalModes=nGlobalModesInitial, verbose=True)
    
    #++++++++++++++++++++++++++++++++++++++++++++++++++
    # iterate over the global modes and remove them successively
    strainEnergySumForPairList = []
    for nGlobalModesUsed in nGlobalModesUsedList: # leave at least two modes
        # initialize beam
        SC = exu.SystemContainer()
        mbs = SC.AddSystem() # always create new mbs (or mbs.Reset())
        nGround = mbs.AddNode(NodePointGround())
        mCoordsGround = mbs.AddMarker(MarkerNodeCoordinates(nodeNumber=nGround, visualization=VMarkerNodeCoordinates(show=False)))
        beamStrainEnergySumPMR = EulerBernulliBeam(beamTemplate=beamTemplateRubber,
                                                   slidingJointFunction=SlidingJointFunctionEvolved,
                                                   loadTipFunction=LoadTipFunctionEvolved)
        
        #++++++++++++++++++++++++++++++++++++++++++++++++++
        # GLOBALLY reduced beam and add it (here we can not use the wrapper for global reduction)
        beamStrainEnergySumPMR.ReduceBeam(reductionBasis=globalReductionBasisInitial[:, 0:nGlobalModesUsed])
        sCoordinatesStrainEnergySumBeamPMR, sCoordinates_tStrainEnergySumBeamPMR = beamStrainEnergySumPMR.AddBeamToMBSAndAddSensors(mbs, mCoordsGround=mCoordsGround, show=False)
        
        #++++++++++++++++++++++++++++++++++++++++++++++++++
        # simulate dynamically (WYSWYS)
        mbs.Assemble()
        simulationSettings = GetStandardSimulationAndSetVisualizationSettings(tEnd, h, exportFrames=False)
        mbs.SolveDynamic(simulationSettings=simulationSettings)
        
        #++++++++++++++++++++++++++++++++++++++++++++++++++
        # get results
        _, time, qStrainEnergySumBeamPMR, q_tStrainEnergySumBeamPMR = beamStrainEnergySumPMR.GetTimeAndCoordinates(mbs, sCoordinatesStrainEnergySumBeamPMR, sCoordinates_tStrainEnergySumBeamPMR)
        strainEnergySum = beamStrainEnergySumPMR.GetSumOfBeamStrainEnergy(qStrainEnergySumBeamPMR)
        strainEnergySumForPairList += [strainEnergySum]
        nAnalysesCompleted += 1
        print(f'Completed {(nAnalysesCompleted/nAnalyses) * 100:05.1f}%.')
    strainEnergySumList += [strainEnergySumForPairList]
AppendResultsToPKLFile(resultsFileName, 'strainEnergySum', {'pairsToBeInvestigatedList': pairsToBeInvestigatedList,
                                                            'nGlobalModesUsedList': nGlobalModesUsedList,
                                                            'strainEnergySumNR': strainEnergySumNR,
                                                            'strainEnergySumList': strainEnergySumList})
#%%
#++++++++++++++++++++++++++++++++++++++++++++++++++
# compute and plot the errors
indexOfMaxNumberOfGlobalModesPlotted = 14
resultsDictStrainEnergySum = GetResultsDictFromPKLFile(resultsFileName, 'strainEnergySum')
pairsToBeInvestigatedList = resultsDictStrainEnergySum['pairsToBeInvestigatedList']
nGlobalModesUsedList = resultsDictStrainEnergySum['nGlobalModesUsedList']
strainEnergySumNR = resultsDictStrainEnergySum['strainEnergySumNR']
strainEnergySumList = resultsDictStrainEnergySum['strainEnergySumList']
nPairsInvestigated = len(pairsToBeInvestigatedList)
nGlobalModesUsed = len(nGlobalModesUsedList)
strainEnergySumErrorMatrix = np.zeros([nPairsInvestigated, nGlobalModesUsed])
for i in range(nPairsInvestigated):
    for j in range(nGlobalModesUsed):
        strainEnergySumErrorMatrix[i, j] = abs(strainEnergySumList[i][j]-strainEnergySumNR)
figStrainSumError, axStrainSumError = plt.subplots(1, 1, figsize=[textWidth, 2.5])
x = np.arange(nGlobalModesUsed-indexOfMaxNumberOfGlobalModesPlotted)
for i, pair in enumerate(pairsToBeInvestigatedList):
    label = r'$n_\mathrm{snap.}=' + str(pair[0]) + r'$, $r_\mathrm{snap.}=' + str(pair[1]) + '$'
    axStrainSumError.plot(x, strainEnergySumErrorMatrix[i, indexOfMaxNumberOfGlobalModesPlotted:]/(1e3), label=label)
axStrainSumError.set_xticks(x, nGlobalModesUsedList[indexOfMaxNumberOfGlobalModesPlotted:])
axStrainSumError.set_xlabel(r'number $r_\mathrm{glob.}$ of global modes used')
axStrainSumError.set_ylabel('abs. strain energy sum error in kJ')
axStrainSumError.legend()
axStrainSumError.grid()

#%% compute global mode sensitivity using strain energy 
PrintHeadlineToConsole(text='Strain Energy Sum Error')
