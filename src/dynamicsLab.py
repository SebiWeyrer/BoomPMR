#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#
# Details:  This is the laboratory script for the 'BoomPMR'-Project.
#           It runs different experiments, where each experiment is abbreviated
#           by a camel-case short-name. The srcipt can write a results file
#           (for all experiments approx. 2.5GB large) or only load results if
#           a results file (.pkl) is placeed in the same directory. Running
#           only one experiment updates this part of the results file. Naming:
#           (1) variable or object names (s..sensor, beam...beam object)
#           (2) the type of experiment abbreviated with short-name
#           (3) formalism identifier if applicable:
#               (3.1) ANCF    Absolute Nodal Coordinate Formulation (ANCF)
#               (3.2) NR      Non-Reduced Euler-Bernoulli (EB) Beam
#               (3.3) PMR     globally Parametric Order Reduced (PMR) EB Beam
#               (3.4) Beam    if used for both NR and PMR
#           (4) the computational object if necessary (e.g. List, PerPairList).
#           For running all experiments, expect long runs (> 3h).
# Author:   Sebastian Weyrer
# Date:     2026-06-05
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import sys
import pickle
import socket
import platform
import numpy as np
import time as tm
import textwrap
import exudyn as exu
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from exudyn.utilities import *
from beams import EulerBernulliBeam, BeamTemplate
from matplotlib.lines import Line2D
from utilities import *
from pathlib import Path
from datetime import datetime
print('EXUDYN says hello with version', exu.config.Version() + '.')

#++++++++++++++++++++++++++++++++++++++++++++++++++
# decide whether only to load results or produce them
onlyLoadResults = True  # if False, expect long execution time due to expensive tests
fastComputation = False # if True and not onlyLoadResults, a fast results file is written to check sanity
saveFigures = False

#++++++++++++++++++++++++++++++++++++++++++++++++++
# default setup of beam reference parameters and one reference beam (that is used for some illustrations, ...)
length = 5          # [m] length of the beams
nElementsANCF = 20  # from convergence analysis (see later)
nElementsBeam = 50  # from convergence analysis (see later)
beamTemplateRubber = BeamTemplate(physicsLength=length,
                                  physicsAxialStiffness=2e7, physicsBendingStiffness=5e4,
                                  physicsMassPerLength=78, nElements=nElementsBeam,
                                  doNotInterpolateForSlidingBC=False)
beamRefNR = EulerBernulliBeam(beamTemplate=beamTemplateRubber, surpressWarningsUponInitialization=True) # also available for fast computation

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
if fastComputation:
    longComputationToggle = 0
else:
    longComputationToggle = 1

#++++++++++++++++++++++++++++++++++++++++++++++++++
# settings to write results file
resultsFileName = 'results.pkl'
if not onlyLoadResults:
    PrintHeadlineToConsole(text='Creating empty results file to save results')
    data = {'metaInfo': {'created': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                         'hostname': socket.gethostname(),
                         'platform': platform.platform(),
                         'pythonVersion': sys.version,
                         'exudynVersion': exu.config.Version()}, 'results': {}}
    with open(resultsFileName, 'wb') as f:
        pickle.dump(data, f)
else:
    try:
        with open(resultsFileName, 'rb') as f:
            data = pickle.load(f)
        PrintHeadlineToConsole(text='Loading Existing Results')
        for info in data['metaInfo']:
            print(textwrap.fill(info + ': ' + data['metaInfo'][info], width=80))
    except:
        print('WARNING: You have to run the script first with "onlyLoadResults=False" to obtain results.')
        sys.exit()
    
#%% plot interpolation of y displacement within beam element
PrintHeadlineToConsole(text='Plot Example Interpolation for Illustration')
q2 = -0.05
q3 = pi/15
q5 = 0.1
q6 = pi/8
def InterpolationOfYDisplacement(alpha):
    N2, N3, N5, N6 = beamRefNR.GetInterpolationCoefficientsForYDisplacement(alpha)
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
eigenmodesLiftedAndMassNormalizedSnapList = beamRefNR.GetGlobalMassNormalizedEigenmodesListForNumberOfConstrainedNodes(nNodesConstrained=3, nEigenmodes=3)
figSnap = beamRefNR.PlotEigenmodesSets(eigenmodesLiftedAndMassNormalizedSnapList, nEigenmodesToPlot=3,
                                              nodeNumbersConstrainedList=beamRefNR.nodeNumbersConstrainedList,
                                              figSize=[textWidth, 4])
SaveFig(figSnap, figPath, 'M_exampleInterpolations', closeFig=True, saveFig=False)

#%% define and plot the two test cases used: SIMPLE and EVOLVED
PrintHeadlineToConsole(text='Define and Plot Test Cases')
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
slidingJointFunctionList = [SlidingJointFunctionSimple, SlidingJointFunctionEvolved]
loadTipFunctionList = [LoadTipFunctionSimple, LoadTipFunctionEvolved]

#++++++++++++++++++++++++++++++++++++++++++++++++++
# plot the two test cases
colorSlider = 'blue'
colorLoad = 'red'
figTc, axTc = plt.subplots(2, 1, figsize=[textWidth, 2.5], sharex=True)
time = np.linspace(0, tEnd, 5001)
# simple test case
axTc[0].plot(time, [SlidingJointFunctionSimple(t) for t in time], '*-', color=colorSlider, markevery=500)
axTc[0].tick_params(axis='y', labelcolor=colorSlider)
axTcLoadSimple = axTc[0].twinx()
axTcLoadSimple.plot(time, [LoadTipFunctionSimple(None, t, None) for t in time], color=colorLoad, alpha=0.75)
axTcLoadSimple.tick_params(axis='y', labelcolor=colorLoad)
axTcLoadSimple.set_ylim([-100, 100])
axTcLoadSimple.set_yticks([-50, 0, 50])
# draw load behind slider
axTcLoadSimple.set_zorder(1)
axTc[0].set_zorder(2)
axTc[0].patch.set_visible(False)
# complex test case
axTc[1].plot(time, [SlidingJointFunctionEvolved(t) for t in time], '*-', color=colorSlider, markevery=500)
axTc[1].tick_params(axis='y', labelcolor=colorSlider)
axTcLoadEvolved = axTc[1].twinx()
axTcLoadEvolved.plot(time, [LoadTipFunctionEvolved(None, t, None) for t in time], color=colorLoad, alpha=0.75)
axTcLoadEvolved.tick_params(axis='y', labelcolor=colorLoad)
axTcLoadEvolved.set_ylim([-100, 100])
axTcLoadEvolved.set_yticks([-50, 0, 50])
# draw load behind slider
axTcLoadEvolved.set_zorder(1)
axTc[1].set_zorder(2)
axTc[1].patch.set_visible(False)
# shared x and y labels
axTc[1].set_xlabel(timeStr)
axTc[1].set_xticks(np.arange(0, 16, 1))
figTc.supylabel(r'$\ast$ slider position $s(t)$ in m', color=colorSlider, x=0.045)
figTc.text(0.97, 0.5, r'tip load $F\,(t)$ in N', color=colorLoad, 
                  va='center', ha='right', rotation=90)
# add grids, shaded area, ...
for i, ax in enumerate(axTc):
    ax.grid()
    if i == 0:
        ax.axvspan(startTime, stopTime, color='gray', alpha=0.2, label=slidingStr, zorder=0)
        ax.legend(loc='upper left')
    else:
        ax.axvspan(startTime, stopTime, color='gray', alpha=0.2, zorder=0)
figTc.tight_layout(rect=[0, 0, 0.96, 1])
SaveFig(figTc, figPath, 'E_testCases', closeFig=False, saveFig=saveFigures)

#%% do convergence analysis to get good amount of ANCF and beam elements
PrintHeadlineToConsole(text='Convergence Analysis Experiment: "Ca"')
if not onlyLoadResults:
    # initialize beam (to inherit some data to ANCF beam)
    beamCaNR = EulerBernulliBeam(beamTemplate=beamTemplateRubber)
    
    #++++++++++++++++++++++++++++++++++++++++++++++++++
    # ANCF elements
    nElementsCaANCFList = np.linspace(2, 44, 20*longComputationToggle+2, dtype=int)
    startTime = tm.time()
    nAnalysesCompleted = 0
    nAnalyses= len(nElementsCaANCFList)
    tipDisplacementCaANCFList = []
    for i, nElementsCaANCF in enumerate(nElementsCaANCFList):
        SC = exu.SystemContainer()
        mbs = SC.AddSystem() # always create new mbs (or mbs.Reset())
        nGround = mbs.AddNode(NodePointGround())
        # this is needed to constrain ANCF beam on the 'left' side
        mCoordGround = mbs.AddMarker(MarkerNodeCoordinate(nodeNumber=nGround, coordinate=0,
                                                          visualization=VMarkerNodeCoordinates(show=False)))
        sensorOutputVariableTypeCaANCFList = [exu.OutputVariableType.Displacement]
        sDictCaANCF = AddANCFBeamWithSlidingJointToMBSAndAddSensors(mbs, SlidingJointFunctionEvolved, LoadTipFunctionEvolved, length=length, rhoA=beamCaNR.rhoA, EI=beamCaNR.EI, EA=beamCaNR.EA,
                                                                    nElementsANCF=nElementsCaANCF, mCoordGround=mCoordGround, sensorOutputVariableTypeANCFList=sensorOutputVariableTypeCaANCFList,
                                                                    beamNodeReferencePositionList=np.array([length])) # only add one sensor in the ANCF approach at tip
        mbs.Assemble()
        simulationSettings = GetStandardSimulationAndSetVisualizationSettings(tEnd, h, exportFrames=False)
        mbs.SolveDynamic(simulationSettings=simulationSettings)
        
        #++++++++++++++++++++++++++++++++++++++++++++++++++
        # get results
        outputDictCaANCF = GetOutputDictANCFFromSDictANCF(mbs, sDictANCF=sDictCaANCF, sensorOutputVariableTypeANCFList=sensorOutputVariableTypeCaANCFList)
        tipDisplacementCaANCFList += [outputDictCaANCF['OutputVariableType.Displacement'][:, -2]]
        nAnalysesCompleted += 1
        ProgressTracker(startTime, nAnalyses, nAnalysesCompleted, showETA=False)
    AppendResultsToPKLFile(resultsFileName, 'Ca', {'nElementsCaANCFList': nElementsCaANCFList,
                                                   'tipDisplacementCaANCFList': tipDisplacementCaANCFList})
    
    #++++++++++++++++++++++++++++++++++++++++++++++++++
    # beam elements
    nElementsCaNRList = np.linspace(2, 60, 28*longComputationToggle+2, dtype=int)
    startTime = tm.time()
    nAnalysesCompleted = 0
    nAnalyses= len(nElementsCaNRList)
    tipDisplacementCaNRList = []
    for i, nElementsCaNR in enumerate(nElementsCaNRList):
        SC = exu.SystemContainer()
        mbs = SC.AddSystem() # always create new mbs (or mbs.Reset())
        nGround = mbs.AddNode(NodePointGround())
        mCoordsGround = mbs.AddMarker(MarkerNodeCoordinates(nodeNumber=nGround, visualization=VMarkerNodeCoordinates(show=False)))
        beamCaNR = EulerBernulliBeam(beamTemplate=beamTemplateRubber,
                                     nElements=nElementsCaNR, # OVERWRITE!
                                     slidingJointFunction=SlidingJointFunctionEvolved,
                                     loadTipFunction=LoadTipFunctionEvolved)
        
        #++++++++++++++++++++++++++++++++++++++++++++++++++
        # add beam to mbs
        sCoordinatesCaNR, sCoordinates_tCaNR = beamCaNR.AddBeamToMBSAndAddSensors(mbs, mCoordsGround=mCoordsGround, show=False)
        
        #++++++++++++++++++++++++++++++++++++++++++++++++++
        # simulate dynamically (WYSWYS)
        mbs.Assemble()
        simulationSettings = GetStandardSimulationAndSetVisualizationSettings(tEnd, h, exportFrames=False)
        mbs.SolveDynamic(simulationSettings=simulationSettings)
        
        #++++++++++++++++++++++++++++++++++++++++++++++++++
        # get results
        _, time, qCaNR, q_tCaNR = beamCaNR.GetTimeAndCoordinates(mbs, sCoordinatesCaNR, sCoordinates_tCaNR)
        tipDisplacementCaNRList += [qCaNR[:, -2]]
        nAnalysesCompleted += 1
        ProgressTracker(startTime, nAnalyses, nAnalysesCompleted, showETA=False)
    AppendResultsToPKLFile(resultsFileName, 'Ca', {'nElementsCaNRList': nElementsCaNRList,
                                                   'tipDisplacementCaNRList': tipDisplacementCaNRList})

#++++++++++++++++++++++++++++++++++++++++++++++++++
# retreive results and compute inetersting quantities
resultsDictCa = GetResultsDictFromPKLFile(resultsFileName, 'Ca')
# ANCF
nElementsCaANCFList = resultsDictCa['nElementsCaANCFList']
tipDisplacementCaANCFList = resultsDictCa['tipDisplacementCaANCFList']
errorCaANCFList = []
for tipDisplacementCaANCF in tipDisplacementCaANCFList[:-1]:
    errorCaANCFList += [AbsError(tipDisplacementCaANCF, tipDisplacementCaANCFList[-1])]
errorCaANCFList += [0]
nElementsCaNRList = resultsDictCa['nElementsCaNRList']
tipDisplacementCaNRList = resultsDictCa['tipDisplacementCaNRList']
errorCaNRList = []
for tipDisplacementCaNR in tipDisplacementCaNRList[:-1]:
    errorCaNRList += [AbsError(tipDisplacementCaNR, tipDisplacementCaNRList[-1])]
errorCaNRList += [0]

#++++++++++++++++++++++++++++++++++++++++++++++++++
# plot
figCa, axCa = plt.subplots(1, 1, figsize=[textWidth, 2.2])
nDOFsANCFList = (nElementsCaANCFList + 1) * 4 # 4 DOfs per node
nDOFsNRList = (nElementsCaNRList + 1) * 3 # 3 DOfs per node
axCa.loglog(nDOFsANCFList[:-1], errorCaANCFList[:-1], '-o', label=labelList[0], color=standardColors[0])
axCa.loglog(nDOFsNRList[:-1], errorCaNRList[:-1], '-*', label=labelList[1], color=standardColors[1])
axCa.grid()
axCa.legend()
axCa.set_xlabel('number of DOFs')
axCa.set_ylabel('abs. error transv.\ntip displacement in m')
SaveFig(figCa, figPath, 'E_convergenceAnalysis', closeFig=False, saveFig=saveFigures)

#%% compute displacement error of globally reduced beam vs. non-reduced beam to obtain reduction error
PrintHeadlineToConsole(text='Reduction Error Experiment: "Re"')
if not onlyLoadResults:
    AppendResultsToPKLFile(resultsFileName, 'Re', {'qRePMRList':[]})
    # determine combinations that should be tested
    nSnapshotsList = np.arange(1, 30+1, 1)
    nEigenmodesPerSnapshotList = np.arange(1, 16+1, 1)
    nMinGlobalModes = 3
    nMaxGlobalModes = 116*longComputationToggle+4
    nGlobalModesEvaluationRange = [16, 20]
    nGlobalModesEvaluation = nGlobalModesEvaluationRange[1] - nGlobalModesEvaluationRange[0] + 1
    nAnalyses = 0
    pairsToInvestigateList = []
    nGlobalModesToInvestigatePerPairList = []
    for nSnapshots in nSnapshotsList:
        for nEigenmodesPerSnapshot in nEigenmodesPerSnapshotList:
            product = nSnapshots * nEigenmodesPerSnapshot
            if product >= nMinGlobalModes and product <= nMaxGlobalModes:
                pairsToInvestigateList += [[nSnapshots, nEigenmodesPerSnapshot]]
                if product >= nGlobalModesEvaluationRange[1]:
                    nGlobalModesToInvestigatePerPairList += [np.arange(nGlobalModesEvaluationRange[0], nGlobalModesEvaluationRange[1]+1, 1)]
                elif product >= nMinGlobalModes + nGlobalModesEvaluation:
                    nGlobalModesToInvestigatePerPairList += [np.arange(product-nGlobalModesEvaluation+1, product+1, 1)]
                elif product >= nMinGlobalModes + 1:
                    nGlobalModesToInvestigatePerPairList += [np.arange(nMinGlobalModes, product+1, 1)]
                elif product == nMinGlobalModes:
                    nGlobalModesToInvestigatePerPairList += [np.array([product])]
                else:
                    print('WARNING: Reduction error computation setup failed!')
                nAnalyses += len(nGlobalModesToInvestigatePerPairList[-1])
    print('Starting a total of ' + str(nAnalyses) + ' analyses.')
    
    #++++++++++++++++++++++++++++++++++++++++++++++++++
    # compute reference solution for NR (in a separate mbs)
    SC = exu.SystemContainer()
    mbs = SC.AddSystem() # always create new mbs (or mbs.Reset())
    nGround = mbs.AddNode(NodePointGround())
    mCoordsGround = mbs.AddMarker(MarkerNodeCoordinates(nodeNumber=nGround, visualization=VMarkerNodeCoordinates(show=False)))
    beamReNR = EulerBernulliBeam(beamTemplate=beamTemplateRubber,
                                  slidingJointFunction=SlidingJointFunctionEvolved,
                                  loadTipFunction=LoadTipFunctionEvolved)
    
    #++++++++++++++++++++++++++++++++++++++++++++++++++
    # add beam to mbs
    sCoordinatesReNR, sCoordinates_tReNR = beamReNR.AddBeamToMBSAndAddSensors(mbs, mCoordsGround=mCoordsGround, show=False)
    
    #++++++++++++++++++++++++++++++++++++++++++++++++++
    # simulate dynamically (WYSWYS)
    mbs.Assemble()
    simulationSettings = GetStandardSimulationAndSetVisualizationSettings(tEnd, h, exportFrames=False)
    mbs.SolveDynamic(simulationSettings=simulationSettings)
    
    #++++++++++++++++++++++++++++++++++++++++++++++++++
    # get results
    _, time, qReNR, q_tReNR = beamReNR.GetTimeAndCoordinates(mbs, sCoordinatesReNR, sCoordinates_tReNR)
    AppendResultsToPKLFile(resultsFileName, 'Re', {'qReNR':qReNR})
    
    #++++++++++++++++++++++++++++++++++++++++++++++++++
    # now process the pairs that should be investigated
    nAnalysesCompleted = 0
    startTime = tm.time()
    for i, pair in enumerate(pairsToInvestigateList):
        nSnapshots = pair[0]
        nEigenmodesPerSnapshot = pair[1]
        qRePMROfPairList = []
        for nGlobalModes in nGlobalModesToInvestigatePerPairList[i]:
            # initialize beam
            SC = exu.SystemContainer()
            mbs = SC.AddSystem() # always create new mbs (or mbs.Reset())
            nGround = mbs.AddNode(NodePointGround())
            mCoordsGround = mbs.AddMarker(MarkerNodeCoordinates(nodeNumber=nGround, visualization=VMarkerNodeCoordinates(show=False)))
            beamRePMR = EulerBernulliBeam(beamTemplate=beamTemplateRubber,
                                          slidingJointFunction=SlidingJointFunctionEvolved,
                                          loadTipFunction=LoadTipFunctionEvolved)
            
            #++++++++++++++++++++++++++++++++++++++++++++++++++
            # GLOBALLY reduced beam and add it
            # we start always with the full globalReductionBasisInitial
            beamRePMR.GlobalReduceBeamWrapper(nSnapshots=nSnapshots, nEigenmodesPerSnapshot=nEigenmodesPerSnapshot,
                                              nGlobalModes=nGlobalModes, usePartialEigensolvers=False) # for certain combos makes difference
            sCoordinatesRePMR, sCoordinates_tRePMR = beamRePMR.AddBeamToMBSAndAddSensors(mbs, mCoordsGround=mCoordsGround, show=False)
            
            #++++++++++++++++++++++++++++++++++++++++++++++++++
            # simulate dynamically (WYSWYS)
            mbs.Assemble()
            simulationSettings = GetStandardSimulationAndSetVisualizationSettings(tEnd, h, exportFrames=False)
            mbs.SolveDynamic(simulationSettings=simulationSettings)
            
            #++++++++++++++++++++++++++++++++++++++++++++++++++
            # get results
            _, time, qRePMR, qRe_tPMR = beamRePMR.GetTimeAndCoordinates(mbs, sCoordinatesRePMR, sCoordinates_tRePMR)
            qRePMROfPairList += [qRePMR]
            nAnalysesCompleted += 1
            ProgressTracker(startTime, nAnalyses, nAnalysesCompleted)
        # to save RAM, we write the results file here already with special append function
        AppendResultsToPKLFile(resultsFileName, 'Re', {'qRePMRList':qRePMROfPairList}, appendList=True)
    AppendResultsToPKLFile(resultsFileName, 'Re', {'nSnapshotsList':nSnapshotsList,
                                                   'nEigenmodesPerSnapshotList':nEigenmodesPerSnapshotList,
                                                   'pairsToInvestigateList': pairsToInvestigateList,
                                                   'nGlobalModesToInvestigatePerPairList': nGlobalModesToInvestigatePerPairList})

#++++++++++++++++++++++++++++++++++++++++++++++++++
# retreive results and compute inetersting quantities
resultsDictRe = GetResultsDictFromPKLFile(resultsFileName, 'Re')
qReNR = resultsDictRe['qReNR']
nSnapshotsList = resultsDictRe['nSnapshotsList']
nEigenmodesPerSnapshotList = resultsDictRe['nEigenmodesPerSnapshotList']
pairsToInvestigateList = resultsDictRe['pairsToInvestigateList']
qRePMRList = resultsDictRe['qRePMRList']
nAnalyses = 0
normDisplacementErrorRePMRList = []
for qRePMROfPairList in qRePMRList:
    normDisplacementErrorRePMROfPairList = []
    for qRePMR in qRePMROfPairList:
        normDisplacementErrorRePMROfPairList += [NRMSE(qRePMR[:, 1::beamRefNR.nDOFsPerNode], qReNR[:, 1::beamRefNR.nDOFsPerNode])]
        nAnalyses += 1
    normDisplacementErrorRePMRList += [np.mean(normDisplacementErrorRePMROfPairList)]
print('Loading a total of', nAnalyses, 'analyses.')
normDisplacementErrorRePMRMatrix = np.full([len(nEigenmodesPerSnapshotList), len(nSnapshotsList)], np.nan)
for i, pair in enumerate(pairsToInvestigateList):
    nSnapshots = pair[0]
    nEigenmodesPerSnapshot = pair[1]
    idxSnapshot = np.where(nSnapshotsList == nSnapshots)
    idxEigenmodesPerSnapshot = np.where(nEigenmodesPerSnapshotList == nEigenmodesPerSnapshot)
    normDisplacementErrorRePMRMatrix[idxEigenmodesPerSnapshot, idxSnapshot] = normDisplacementErrorRePMRList[i]

#++++++++++++++++++++++++++++++++++++++++++++++++++
# plot
figRe, axRe = plt.subplots(1, 1, figsize=[textWidth, 2.5])
im = axRe.imshow(normDisplacementErrorRePMRMatrix, cmap='RdYlGn_r', vmin=0, vmax=1)
xTickIndices = np.round(np.linspace(0, len(nSnapshotsList) - 1, 5)).astype(int)
yTickIndices = np.round(np.linspace(0, len(nEigenmodesPerSnapshotList) - 1, 5)).astype(int)
axRe.set_xticks(xTickIndices)
axRe.set_xticklabels(nSnapshotsList[xTickIndices])
axRe.set_yticks(yTickIndices)
axRe.set_yticklabels(nEigenmodesPerSnapshotList[yTickIndices])
axRe.invert_yaxis()
axRe.set_xlabel(r'number $n_\mathrm{snap.}$ of snapshots')
axRe.set_ylabel(r'number $r_\mathrm{snap.}$ of' + '\neigenmodes per snapshot')
plt.colorbar(im, label='norm. RMS error transv.\nbeam displacement')
# contour
contourLevel = 26
xi = np.linspace(0, len(nSnapshotsList) - 1, 500)
yi = np.linspace(0, len(nEigenmodesPerSnapshotList) - 1, 500)
XI, YI = np.meshgrid(xi, yi)
snapVals = np.interp(xi, np.arange(len(nSnapshotsList)), nSnapshotsList)
eigenVals = np.interp(yi, np.arange(len(nEigenmodesPerSnapshotList)), nEigenmodesPerSnapshotList)
SNAP, EIGEN = np.meshgrid(snapVals, eigenVals)
cs = axRe.contour(XI, YI, SNAP * EIGEN, levels=[contourLevel], colors='black', linewidths=2)
# rectangles to show nLowest combos
nLowest = 15
flatIdx = np.argsort(normDisplacementErrorRePMRMatrix, axis=None)[:nLowest]
rows, cols = np.unravel_index(flatIdx, normDisplacementErrorRePMRMatrix.shape)
for r, c in zip(rows, cols):
    axRe.add_patch(mpatches.Rectangle((c - 0.5, r - 0.5), 1, 1, linewidth=1.5, edgecolor='black', facecolor='none', zorder=5))
legendHandles = [Line2D([0], [0], color='black', linewidth=2, label=r'$n_\mathrm{snap.} \, r_\mathrm{snap.} = ' + str(contourLevel) + '$'),
                 mpatches.Rectangle((0, 0), 1, 1, linewidth=1.5, edgecolor='black', facecolor='none', label=str(nLowest) + ' lowest errors'),]
axRe.legend(handles=legendHandles, loc='upper right')
SaveFig(figRe, figPath, 'E_reductionError', closeFig=False, saveFig=saveFigures)

#++++++++++++++++++++++++++++++++++++++++++++++++++
# get best combination for treated problem
bestComboIndices = np.unravel_index(np.nanargmin(normDisplacementErrorRePMRMatrix), normDisplacementErrorRePMRMatrix.shape)
nSnapshotsBest = nSnapshotsList[bestComboIndices[1]]
nEigenmodesPerSnapshotBest = nEigenmodesPerSnapshotList[bestComboIndices[0]]
print('Best combo: nSapshots = ' + str(nSnapshotsBest) + '; nEigenmodesPerSnapshot = ' + str(nEigenmodesPerSnapshotBest))

#%% compute quantities for specific selected combos
PrintHeadlineToConsole(text='Selected Combos Experiment: "Sc"')
if not onlyLoadResults:
    AppendResultsToPKLFile(resultsFileName, 'Sc', {'qScPMRList':[], 'q_tScPMRList':[]})
    AppendResultsToPKLFile(resultsFileName, 'Sc', {'sList':[]})
    # determine combinations that should be tested
    if fastComputation:
        pairsToInvestigateList = [(4, 1), (2, 4)]
    else:
        pairsToInvestigateList = [(nSnapshotsBest, nEigenmodesPerSnapshotBest), (10, 10), (15, 1)] # nSnapshots and nEigenmodePerSnapshot
    nMaxGlobalModesToInvestigate = 35
    nAnalyses = 0
    nGlobalModesToInvestigatePerPairList = []
    for pair in pairsToInvestigateList:
        if pair[0] * pair[1] > nMaxGlobalModesToInvestigate:
            nGlobalModesInitial = nMaxGlobalModesToInvestigate
        else:
            nGlobalModesInitial = pair[0] * pair[1]
        nGlobalModesToInvestigatePerPairList += [np.flipud(np.arange(3, nGlobalModesInitial+1, 1))]
        nAnalyses += len(nGlobalModesToInvestigatePerPairList[-1])
    
    #++++++++++++++++++++++++++++++++++++++++++++++++++
    # now process the pairs that should be investigated
    nAnalysesCompleted = 0
    startTime = tm.time()
    for i, pair in enumerate(pairsToInvestigateList):
        # offline combute initial reduction basis for the investigated pairs
        beamScPMR = EulerBernulliBeam(beamTemplate=beamTemplateRubber)
        eigenmodesLiftedAndMassNormalizedList = beamScPMR.GetGlobalMassNormalizedEigenmodesListForNumberOfConstrainedNodes(nNodesConstrained=pair[0], nEigenmodes=pair[1],
                                                                                                                           usePartialEigensolvers=False) # for certain combos makes difference
        globalReductionBasisInitial, S = beamScPMR.GetGlobalReductionBasisUsingEigenmodesList(eigenmodesList=eigenmodesLiftedAndMassNormalizedList,
                                                                                              nGlobalModes=nGlobalModesToInvestigatePerPairList[i][0],
                                                                                              returnSingularValues=True)
        AppendResultsToPKLFile(resultsFileName, 'Sc', {'sList':S}, appendList=True)
        
        #++++++++++++++++++++++++++++++++++++++++++++++++++
        # iterate over the global modes and remove them successively
        qScPMRForPairList = []
        q_tScPMRForPairList = []
        for nGlobalModes in nGlobalModesToInvestigatePerPairList[i]:
            # initialize beam
            SC = exu.SystemContainer()
            mbs = SC.AddSystem() # always create new mbs (or mbs.Reset())
            nGround = mbs.AddNode(NodePointGround())
            mCoordsGround = mbs.AddMarker(MarkerNodeCoordinates(nodeNumber=nGround, visualization=VMarkerNodeCoordinates(show=False)))
            beamScPMR = EulerBernulliBeam(beamTemplate=beamTemplateRubber,
                                          slidingJointFunction=SlidingJointFunctionEvolved,
                                          loadTipFunction=LoadTipFunctionEvolved)
            
            #++++++++++++++++++++++++++++++++++++++++++++++++++
            # GLOBALLY reduced beam and add it (here we can not use the wrapper for global reduction)
            # we start always with the full globalReductionBasisInitial
            beamScPMR.ReduceBeam(reductionBasis=globalReductionBasisInitial[:, 0:nGlobalModes])
            sCoordinatesScPMR, sCoordinates_tScPMR = beamScPMR.AddBeamToMBSAndAddSensors(mbs, mCoordsGround=mCoordsGround, show=False)
            
            #++++++++++++++++++++++++++++++++++++++++++++++++++
            # simulate dynamically (WYSWYS)
            mbs.Assemble()
            simulationSettings = GetStandardSimulationAndSetVisualizationSettings(tEnd, h, exportFrames=False)
            mbs.SolveDynamic(simulationSettings=simulationSettings)
            
            #++++++++++++++++++++++++++++++++++++++++++++++++++
            # get results
            _, time, qScPMR, q_tScPMR = beamScPMR.GetTimeAndCoordinates(mbs, sCoordinatesScPMR, sCoordinates_tScPMR)
            qScPMRForPairList += [qScPMR]
            q_tScPMRForPairList += [q_tScPMR]
            nAnalysesCompleted += 1
            ProgressTracker(startTime, nAnalyses, nAnalysesCompleted)
        # to save RAM, we write the results file here already with special append function
        AppendResultsToPKLFile(resultsFileName, 'Sc', {'qScPMRList':qScPMRForPairList,
                                                       'q_tScPMRList':q_tScPMRForPairList}, appendList=True)
    AppendResultsToPKLFile(resultsFileName, 'Sc', {'pairsToInvestigateList': pairsToInvestigateList,
                                                   'nGlobalModesToInvestigatePerPairList': nGlobalModesToInvestigatePerPairList})

#++++++++++++++++++++++++++++++++++++++++++++++++++
# retreive results and compute inetersting quantities
resultsDictSc = GetResultsDictFromPKLFile(resultsFileName, 'Sc')
pairsToInvestigateList = resultsDictSc['pairsToInvestigateList']
nGlobalModesToInvestigatePerPairList = resultsDictSc['nGlobalModesToInvestigatePerPairList']
qScPMRList = resultsDictSc['qScPMRList']
sList = resultsDictSc['sList']
normDisplacementErrorScPMRList = []
strainSumErrorScPMRList = []
strainSumScNR = beamRefNR.GetSumOfBeamStrainEnergy(qReNR)
for qScPMROfPairList in qScPMRList:
    normDisplacementErrorScPMROfPairList = []
    strainSumErrorScPMROfPairList = []
    for qScPMR in qScPMROfPairList:
        normDisplacementErrorScPMROfPairList += [min(NRMSE(qScPMR[:, 1::beamRefNR.nDOFsPerNode], qReNR[:, 1::beamRefNR.nDOFsPerNode]), 1)]
        strainSumErrorScPMROfPairList += [abs(strainSumScNR-beamRefNR.GetSumOfBeamStrainEnergy(qScPMR))]
    normDisplacementErrorScPMRList += [normDisplacementErrorScPMROfPairList]
    strainSumErrorScPMRList += [strainSumErrorScPMROfPairList]

#++++++++++++++++++++++++++++++++++++++++++++++++++
# plot
figErrorSc, axErrorSc = plt.subplots(2, 1, figsize=[textWidth, 3.1], sharex=True)
nGlobalModesMax = 0
for i, (normDisplacementErrorScPMROfPairList, strainSumErrorScPMROfPairList) in enumerate(zip(normDisplacementErrorScPMRList, strainSumErrorScPMRList)):
    label = r'$n_\mathrm{snap.}=' + str(pairsToInvestigateList[i][0]) + r'$, $r_\mathrm{snap.}=' + str(pairsToInvestigateList[i][1]) + r'$'
    x = nGlobalModesToInvestigatePerPairList[i]  # actual mode numbers
    if max(x) > nGlobalModesMax:
        nGlobalModesMax = max(x)
    axErrorSc[0].semilogy(x, normDisplacementErrorScPMROfPairList, markerList[i], label=label)
    axErrorSc[1].semilogy(x, strainSumErrorScPMROfPairList, markerList[i], label=label)
for i, normDisplacementErrorScPMROfPairList in enumerate(normDisplacementErrorScPMRList):
    x = np.array(nGlobalModesToInvestigatePerPairList[i])
    y = np.array(normDisplacementErrorScPMROfPairList)
    xMax = min(max(x), 20)
    xShadeStart = xMax - 4
    mask = (x >= xShadeStart) & (x <= xMax)
    xShaded = x[mask]
    yShaded = y[mask]
    if i == 0:
        axErrorSc[0].fill_between(xShaded, 0, yShaded, alpha=0.3, color=f'C{i}')
    else:
        axErrorSc[0].fill_between(xShaded, 0, yShaded, alpha=0.2, color=f'C{i}')
for ax in axErrorSc:
    ax.grid()
# axErrorSc[0].invert_xaxis()
axErrorSc[1].legend()
axErrorSc[0].set_xticks(np.arange(3, nGlobalModesMax, 3))
axErrorSc[0].set_ylabel('norm. RMS error transv.\nbeam displacement')
axErrorSc[1].set_ylabel('abs. error\nstrain sum in J')
axErrorSc[1].set_xlabel(r'number $r_\mathrm{glob.}$ of global modes in global reduction basis')
axErrorSc[1].set_xlim([2, nGlobalModesMax])
figErrorSc.tight_layout(rect=[0, 0, 1, 1.1]) # in pdf it must be rescaled a bit due to long label
SaveFig(figErrorSc, figPath, 'E_selectedCombosError', closeFig=False, saveFig=saveFigures)
figSingularValuesSc, axSingularValuesSc = plt.subplots(1, 1, figsize=[textWidth, 2.2])
nMaxSingularValuesToPlot = 100
for i, s in enumerate(sList):
    if nMaxSingularValuesToPlot > len(s):
        nSingularValuesToPlot = len(s)
    else:
        nSingularValuesToPlot = nMaxSingularValuesToPlot
    label = r'$n_\mathrm{snap.}=' + str(pairsToInvestigateList[i][0]) + r'$, $r_\mathrm{snap.}=' + str(pairsToInvestigateList[i][1]) + r'$'
    # axSingularValuesSc[0].plot(range(len(s))[:nSingularValuesToPlot], s[:nSingularValuesToPlot], markerList[i], label=label)
    axSingularValuesSc.semilogy(range(len(s))[:nSingularValuesToPlot], s[:nSingularValuesToPlot], markerList[i], label=label,
                                markevery=2)
    # sMax = s[:nSingularValuesToPlot][0]
    # sMin = s[:nSingularValuesToPlot][-1]
    # sNormalized = [((s[:nSingularValuesToPlot][i]-sMin)/(sMax-sMin)) for i in range(nSingularValuesToPlot)]
    # axSingularValuesSc[1].plot(range(len(s))[:nSingularValuesToPlot], sNormalized, markerList[i], label=label)
axSingularValuesSc.grid()
axSingularValuesSc.set_ylabel('singular value')
# labelX = -0.11  # tune this value
# axSingularValuesSc[0].yaxis.set_label_coords(labelX, 0.5)
# axSingularValuesSc[1].yaxis.set_label_coords(labelX, 0.5)
axSingularValuesSc.legend()
axSingularValuesSc.set_xlabel('global mode')
SaveFig(figSingularValuesSc, figPath, 'E_selectedCombosSingularValues', closeFig=False, saveFig=saveFigures)

#++++++++++++++++++++++++++++++++++++++++++++++++++
# compute kappa
nGlobalModes = 20
for i, pair in enumerate(pairsToInvestigateList):
    nSnapshots = pair[0]
    nEigenmodesPerSnapshot = pair[1]
    num = 0
    denum = 0
    for j, sigma in enumerate(sList[i]):
        if j < 20:
            num += sigma**2
        denum += sigma**2
    kappa = num / denum
    print(f'({nSnapshots}, {nEigenmodesPerSnapshot}): kappa={kappa}')

#%% Hamiltonian for sliding phase and numerical results
PrintHeadlineToConsole(text='Hamiltonian and Numerical Results Experiment: "HNum"')
if not onlyLoadResults:
    nGlobalModes = 20
    AppendResultsToPKLFile(resultsFileName, 'HNum', {'outputDictHNumANCFPerTestCaseList':[],
                                                     'qHNumNRPerTestCaseList':[],
                                                     'q_tHNumNRPerTestCaseList':[],
                                                     'qHNumPMRPerTestCaseList':[],
                                                     'q_tHNumPMRPerTestCaseList':[],
                                                     'EHNumANCFPerTestCaseList':[],
                                                     'EHNumNRPerTestCaseList':[],
                                                     'EHNumPMRPerTestCaseList':[]})
    for i in range(2):
        SC = exu.SystemContainer()
        mbs = SC.AddSystem() # always create new mbs (or mbs.Reset())
        nGround = mbs.AddNode(NodePointGround())
        mCoordsGround = mbs.AddMarker(MarkerNodeCoordinates(nodeNumber=nGround, visualization=VMarkerNodeCoordinates(show=False)))
        mCoordGround = mbs.AddMarker(MarkerNodeCoordinate(nodeNumber=nGround, coordinate=0,
                                                          visualization=VMarkerNodeCoordinates(show=False)))
        
        #++++++++++++++++++++++++++++++++++++++++++++++++++
        # ANCF
        sensorOutputVariableTypeHNumANCFList = [exu.OutputVariableType.Displacement,
                                                exu.OutputVariableType.Rotation,
                                                exu.OutputVariableType.Velocity,
                                                exu.OutputVariableType.AngularVelocity,
                                                exu.OutputVariableType.StrainLocal,
                                                exu.OutputVariableType.CurvatureLocal,
                                                exu.OutputVariableType.ForceLocal,
                                                exu.OutputVariableType.TorqueLocal]
        sDictHNumANCF = AddANCFBeamWithSlidingJointToMBSAndAddSensors(mbs, slidingJointFunctionList[i], loadTipFunctionList[i], length=length, rhoA=beamRefNR.rhoA, EI=beamRefNR.EI, EA=beamRefNR.EA,
                                                                      nElementsANCF=nElementsANCF, mCoordGround=mCoordGround, sensorOutputVariableTypeANCFList=sensorOutputVariableTypeHNumANCFList,
                                                                      beamNodeReferencePositionList=beamRefNR.nodeReferencePositionsList)
    
        #++++++++++++++++++++++++++++++++++++++++++++++++++
        # NR
        beamHNumNR = EulerBernulliBeam(beamTemplate=beamTemplateRubber,
                                       slidingJointFunction=slidingJointFunctionList[i],
                                       loadTipFunction=loadTipFunctionList[i])
        sCoordinatesHNumNR, sCoordinates_tHNumNR = beamHNumNR.AddBeamToMBSAndAddSensors(mbs, mCoordsGround=mCoordsGround, show=False)
        
        #++++++++++++++++++++++++++++++++++++++++++++++++++
        # PMR
        beamHNumPMR = EulerBernulliBeam(beamTemplate=beamTemplateRubber,
                                        slidingJointFunction=slidingJointFunctionList[i],
                                        loadTipFunction=loadTipFunctionList[i])
        beamHNumPMR.GlobalReduceBeamWrapper(nGlobalModes=nGlobalModes, nSnapshots=nSnapshotsBest, nEigenmodesPerSnapshot=nEigenmodesPerSnapshotBest)
        sCoordinatesHNumPMR, sCoordinates_tHNumPMR = beamHNumPMR.AddBeamToMBSAndAddSensors(mbs, mCoordsGround=mCoordsGround, show=False)
        
        
        
        #++++++++++++++++++++++++++++++++++++++++++++++++++
        # simulate dynamically (WYSWYS)
        mbs.Assemble()
        simulationSettings = GetStandardSimulationAndSetVisualizationSettings(tEnd, h, exportFrames=False)
        mbs.SolveDynamic(simulationSettings=simulationSettings)
        
        #++++++++++++++++++++++++++++++++++++++++++++++++++
        # get results
        outputDictHNumANCF = GetOutputDictANCFFromSDictANCF(mbs, sDictANCF=sDictHNumANCF, sensorOutputVariableTypeANCFList=sensorOutputVariableTypeHNumANCFList)
        nTimeSteps, time, qHNumNR, q_tHNumNR = beamHNumNR.GetTimeAndCoordinates(mbs, sCoordinatesHNumNR, sCoordinates_tHNumNR)
        _, time, qHNumPMR, q_tHNumPMR = beamHNumPMR.GetTimeAndCoordinates(mbs, sCoordinatesHNumPMR, sCoordinates_tHNumPMR)
        # compute the energies
        EHNumANCF = GetKineticAndStrainAndPotentialANCFEnergyFromOutputDictANCF(outputDictHNumANCF, nTimeSteps=nTimeSteps, rhoA=beamRefNR.rhoA, LoadTipFunction=loadTipFunctionList[i],
                                                                                length=length, nBeamElements=nElementsBeam, sensorsWritePeriod=simulationSettings.solutionSettings.sensorsWritePeriod)
        EHNumNR = beamHNumNR.GetKineticAndStrainAndPotentialBeamEnergy(qHNumNR, q_tHNumNR, sensorWritePeriod=simulationSettings.solutionSettings.sensorsWritePeriod)
        EHNumPMR = beamHNumPMR.GetKineticAndStrainAndPotentialBeamEnergy(qHNumPMR, q_tHNumPMR, sensorWritePeriod=simulationSettings.solutionSettings.sensorsWritePeriod)
        AppendResultsToPKLFile(resultsFileName, 'HNum', {'outputDictHNumANCFPerTestCaseList':outputDictHNumANCF,
                                                         'qHNumNRPerTestCaseList':qHNumNR,
                                                         'q_tHNumNRPerTestCaseList':q_tHNumNR,
                                                         'qHNumPMRPerTestCaseList':qHNumPMR,
                                                         'q_tHNumPMRPerTestCaseList':q_tHNumPMR,
                                                         'EHNumANCFPerTestCaseList':EHNumANCF,
                                                         'EHNumNRPerTestCaseList':EHNumNR,
                                                         'EHNumPMRPerTestCaseList':EHNumPMR}, appendList=True)
    AppendResultsToPKLFile(resultsFileName, 'HNum', {'time':time, 'nTimeSteps':nTimeSteps}) # same for all

#++++++++++++++++++++++++++++++++++++++++++++++++++
# retreive results and compute inetersting quantities
resultsDictHNum = GetResultsDictFromPKLFile(resultsFileName, 'HNum')
outputDictHNumANCFPerTestCaseList = resultsDictHNum['outputDictHNumANCFPerTestCaseList']
qHNumNRPerTestCaseList = resultsDictHNum['qHNumNRPerTestCaseList']
q_tHNumNRPerTestCaseList = resultsDictHNum['q_tHNumNRPerTestCaseList']
qHNumPMRPerTestCaseList = resultsDictHNum['qHNumPMRPerTestCaseList']
q_tHNumPMRPerTestCaseList = resultsDictHNum['q_tHNumPMRPerTestCaseList']
EHNumANCFPerTestCaseList = resultsDictHNum['EHNumANCFPerTestCaseList']
EHNumNRPerTestCaseList = resultsDictHNum['EHNumNRPerTestCaseList']
EHNumPMRPerTestCaseList = resultsDictHNum['EHNumPMRPerTestCaseList']
time = resultsDictHNum['time']

#++++++++++++++++++++++++++++++++++++++++++++++++++
# make table with numerical results
print('=====================================================')
for i in range(2):
    if i == 0:
        print('Results Test Case "Simple"')
    elif i == 1:
        print('=====================================================')
        print('Results Test Case "Evolved"')
    displacementANCF = outputDictHNumANCFPerTestCaseList[i]['OutputVariableType.Displacement']
    qHNumNR = qHNumNRPerTestCaseList[i]
    qHNumPMR = qHNumPMRPerTestCaseList[i]
    displacementErrorNRANCF = NRMSE(qHNumNR[:, 1::beamRefNR.nDOFsPerNode], displacementANCF[:, 1::3])
    print('norm. Beam Displacement; NR against ANCF:\t', round(displacementErrorNRANCF, 4))
    displacementErrorPMRANCF = NRMSE(qHNumPMR[:, 1::beamRefNR.nDOFsPerNode], displacementANCF[:, 1::3])
    print('norm. Beam Displacement; PMR against ANCF:\t', round(displacementErrorPMRANCF, 4))
    displacementErrorPMRNR = NRMSE(qHNumPMR[:, 1::beamRefNR.nDOFsPerNode], qHNumNR[:, 1::beamRefNR.nDOFsPerNode])
    print('norm. Beam Displacement; PMR against NR:\t', round(displacementErrorPMRNR, 4))
    tipDisplacementErrorNRANCF = NRMSE(qHNumNR[:, -2], displacementANCF[:, -2])
    print('norm. Tip Displacement; NR against ANCF:\t', round(tipDisplacementErrorNRANCF, 4))
    tipDisplacementErrorPMRANCF = NRMSE(qHNumPMR[:, -2], displacementANCF[:, -2])
    print('norm. Tip Displacement; PMR against ANCF:\t', round(tipDisplacementErrorPMRANCF, 4))
    tipDisplacementErrorPMRNR = NRMSE(qHNumPMR[:, -2], qHNumNR[:, -2])
    print('norm. Tip Displacement; PMR against NR:\t\t', round(tipDisplacementErrorPMRNR, 4))
print('=====================================================')

#++++++++++++++++++++++++++++++++++++++++++++++++++
# plot
for i in range(2):
    figTipDisplacementHamiltonianHNum, axTipDisplacementHamiltonianHNum = plt.subplots(2, 1, figsize=[textWidth, 3.1], sharex=True)
    markerInterval = 50
    markerIntervalHamiltonianPlotFactor = 4
    markEvery = (int(0*markerInterval/3), markerInterval)
    axTipDisplacementHamiltonianHNum[0].plot(time, outputDictHNumANCFPerTestCaseList[i]['OutputVariableType.Displacement'][:, -2], markerList[0], markevery=markEvery,
                                  label=labelList[0])
    markEvery = (int(0*markerInterval*markerIntervalHamiltonianPlotFactor/3), markerInterval*markerIntervalHamiltonianPlotFactor)
    axTipDisplacementHamiltonianHNum[1].plot(time, EHNumANCFPerTestCaseList[i], markerList[0], markevery=markEvery,
                                  label=labelList[0])
    markEvery = (int(1*markerInterval/3), markerInterval)
    axTipDisplacementHamiltonianHNum[0].plot(time, qHNumNRPerTestCaseList[i][:, -2], markerList[1], markevery=markEvery,
                                  label=labelList[1])
    markEvery = (int(1*markerInterval*markerIntervalHamiltonianPlotFactor/3), markerInterval*markerIntervalHamiltonianPlotFactor)
    axTipDisplacementHamiltonianHNum[1].plot(time, EHNumNRPerTestCaseList[i], markerList[1], markevery=markEvery,
                                  label=labelList[1])
    markEvery = (int(2*markerInterval/3), markerInterval)
    axTipDisplacementHamiltonianHNum[0].plot(time, qHNumPMRPerTestCaseList[i][:, -2], markerList[2], markevery=markEvery,
                                  label=labelList[2])
    markEvery = (int(2*markerInterval*markerIntervalHamiltonianPlotFactor/3), markerInterval*markerIntervalHamiltonianPlotFactor)
    axTipDisplacementHamiltonianHNum[1].plot(time, EHNumPMRPerTestCaseList[i], markerList[2], markevery=markEvery,
                                  label=labelList[2])
    axTipDisplacementHamiltonianHNum[0].set_ylabel('transv.\ntip displacement in m')
    axTipDisplacementHamiltonianHNum[1].set_ylabel('Hamiltonian of\nthe system in J')
    labelX = -0.13  # tune this value
    axTipDisplacementHamiltonianHNum[0].yaxis.set_label_coords(labelX, 0.5)
    axTipDisplacementHamiltonianHNum[1].yaxis.set_label_coords(labelX, 0.5)
    for ax in axTipDisplacementHamiltonianHNum:
        ax.grid()
        ax.axvspan(startTime, stopTime, color='gray', alpha=0.2, label=slidingStr)
    axTipDisplacementHamiltonianHNum[1].legend()
    axTipDisplacementHamiltonianHNum[1].set_xticks(np.arange(0, time[-1]+1, 3))
    axTipDisplacementHamiltonianHNum[1].set_xlabel(timeStr)
    if i == 0:
        SaveFig(figTipDisplacementHamiltonianHNum, figPath, 'E_hamiltonianTipDisplacementSimple', closeFig=False, saveFig=saveFigures)
    elif i == 1:
        SaveFig(figTipDisplacementHamiltonianHNum, figPath, 'E_hamiltonianTipDisplacementEvolved', closeFig=False, saveFig=saveFigures)

#%% computational speed
PrintHeadlineToConsole(text='Computational Speed Experiment: "Cs"')
if not onlyLoadResults:
    nCases = 5
    nGlobalModes = 20
    nAnalyses = nCases * 2
    nAnalysesCompleted = 0
    startTime = tm.time()
    computationTimeANCFPerTestCaseList = []
    computationTimeNRPerTestCaseList = []
    computationTimePMRPerTestCaseList = []
    for i in range(2):
        computationTimeANCFOfTestCaseList = []
        computationTimeNROfTestCaseList = []
        computationTimePMROfTestCaseList = []
        for _ in range(nCases): # now always create separate mbs!
            # ANCF
            SC = exu.SystemContainer()
            mbs = SC.AddSystem() # always create new mbs (or mbs.Reset())
            nGround = mbs.AddNode(NodePointGround())
            mCoordGround = mbs.AddMarker(MarkerNodeCoordinate(nodeNumber=nGround, coordinate=0,
                                                      visualization=VMarkerNodeCoordinates(show=False)))
            sensorOutputVariableTypeCsANCFList = [exu.OutputVariableType.Displacement,
                                                    exu.OutputVariableType.Velocity]
            _ = AddANCFBeamWithSlidingJointToMBSAndAddSensors(mbs, slidingJointFunctionList[i], loadTipFunctionList[i], length=length, rhoA=beamRefNR.rhoA, EI=beamRefNR.EI, EA=beamRefNR.EA,
                                                              nElementsANCF=nElementsANCF, mCoordGround=mCoordGround, sensorOutputVariableTypeANCFList=sensorOutputVariableTypeCsANCFList,
                                                              beamNodeReferencePositionList=beamRefNR.nodeReferencePositionsList)
            mbs.Assemble()
            simulationSettings = GetStandardSimulationAndSetVisualizationSettings(tEnd, h, exportFrames=False)
            specificStartTime = tm.time()
            mbs.SolveDynamic(simulationSettings=simulationSettings)
            computationTimeANCFOfTestCaseList += [tm.time()-specificStartTime]

            #++++++++++++++++++++++++++++++++++++++++++++++++++
            # NR
            SC = exu.SystemContainer()
            mbs = SC.AddSystem() # always create new mbs (or mbs.Reset())
            nGround = mbs.AddNode(NodePointGround())
            mCoordsGround = mbs.AddMarker(MarkerNodeCoordinates(nodeNumber=nGround, visualization=VMarkerNodeCoordinates(show=False)))
            beamCsNR = EulerBernulliBeam(beamTemplate=beamTemplateRubber,
                                         slidingJointFunction=slidingJointFunctionList[i],
                                         loadTipFunction=loadTipFunctionList[i])
            _, _ = beamCsNR.AddBeamToMBSAndAddSensors(mbs, mCoordsGround=mCoordsGround, show=False)
            mbs.Assemble()
            simulationSettings = GetStandardSimulationAndSetVisualizationSettings(tEnd, h, exportFrames=False)
            specificStartTime = tm.time()
            mbs.SolveDynamic(simulationSettings=simulationSettings)
            computationTimeNROfTestCaseList += [tm.time()-specificStartTime]
    
            #++++++++++++++++++++++++++++++++++++++++++++++++++
            # PMR
            SC = exu.SystemContainer()
            mbs = SC.AddSystem() # always create new mbs (or mbs.Reset())
            nGround = mbs.AddNode(NodePointGround())
            mCoordsGround = mbs.AddMarker(MarkerNodeCoordinates(nodeNumber=nGround, visualization=VMarkerNodeCoordinates(show=False)))
            beamCsPMR = EulerBernulliBeam(beamTemplate=beamTemplateRubber,
                                          slidingJointFunction=slidingJointFunctionList[i],
                                          loadTipFunction=loadTipFunctionList[i])
            beamCsPMR.GlobalReduceBeamWrapper(nGlobalModes=nGlobalModes, nSnapshots=nSnapshotsBest, nEigenmodesPerSnapshot=nEigenmodesPerSnapshotBest)
            _, _ = beamCsPMR.AddBeamToMBSAndAddSensors(mbs, mCoordsGround=mCoordsGround, show=False)
            mbs.Assemble()
            simulationSettings = GetStandardSimulationAndSetVisualizationSettings(tEnd, h, exportFrames=False)
            specificStartTime = tm.time()
            mbs.SolveDynamic(simulationSettings=simulationSettings)
            computationTimePMROfTestCaseList += [tm.time()-specificStartTime]
            
            #++++++++++++++++++++++++++++++++++++++++++++++++++
            # not getting results but track progress
            nAnalysesCompleted += 1
            ProgressTracker(startTime, nAnalyses, nAnalysesCompleted)
        computationTimeANCFPerTestCaseList += [computationTimeANCFOfTestCaseList]
        computationTimeNRPerTestCaseList += [computationTimeNROfTestCaseList]
        computationTimePMRPerTestCaseList += [computationTimePMROfTestCaseList]
    AppendResultsToPKLFile(resultsFileName, 'Cs', {'computationTimeANCFPerTestCaseList':computationTimeANCFPerTestCaseList,
                                                   'computationTimeNRPerTestCaseList':computationTimeNRPerTestCaseList,
                                                   'computationTimePMRPerTestCaseList':computationTimePMRPerTestCaseList})
    
#++++++++++++++++++++++++++++++++++++++++++++++++++
# retreive results and compute inetersting quantities
resultsDictCs = GetResultsDictFromPKLFile(resultsFileName, 'Cs')
computationTimeANCFPerTestCaseList = resultsDictCs['computationTimeANCFPerTestCaseList']
computationTimeNRPerTestCaseList = resultsDictCs['computationTimeNRPerTestCaseList']
computationTimePMRPerTestCaseList = resultsDictCs['computationTimePMRPerTestCaseList']

#++++++++++++++++++++++++++++++++++++++++++++++++++
# plot
labels = labelList[0:3]
dataList = [computationTimeANCFPerTestCaseList, computationTimeNRPerTestCaseList, computationTimePMRPerTestCaseList]
meanComputationTimeSimpleCs = []
meanComputationTimeEvolvedCs = []
stdComputationTimeSimpleCs = []
stdComputationTimeEvolvedCs = []
for dataPerTestCaseList in dataList:
    meanComputationTimeSimpleCs += [np.mean(dataPerTestCaseList[0])]
    meanComputationTimeEvolvedCs += [np.mean(dataPerTestCaseList[1])]
    stdComputationTimeSimpleCs += [np.std(dataPerTestCaseList[0])]
    stdComputationTimeEvolvedCs += [np.std(dataPerTestCaseList[1])]
meanComputationTimeCs = [meanComputationTimeSimpleCs, meanComputationTimeEvolvedCs]
stdComputationTimeCs = [stdComputationTimeSimpleCs, stdComputationTimeEvolvedCs]
x = np.arange(len(labels))
width = 0.4
multiplier = 0
figCs, axCs = plt.subplots(1, 1, figsize=[textWidth, 2])
for i in range(2):
    offset = width * multiplier
    if i == 0:
        label = r'simple'
    else:
        label = r'evolved'
    rects = axCs.bar(x + offset, meanComputationTimeCs[i], width, label=label,
                     # yerr=stdComputationTimeCs[i], capsize=5
                     )
    axCs.bar_label(rects, padding=3, fmt='%.2f')
    multiplier += 1
axCs.set_yscale('log')
axCs.set_ylim(top=60)
axCs.legend()
axCs.set_xticks(x+width/2)
axCs.set_xticklabels(labels)
axCs.set_ylabel('computation time in s')
SaveFig(figCs, figPath, 'E_computationSpeed', closeFig=False, saveFig=saveFigures)
