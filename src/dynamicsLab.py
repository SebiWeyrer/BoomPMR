#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#
# Details:  
# Author:   Sebastian Weyrer
# Date:     2026-04-01
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import os
import time as tm
import contextlib
import numpy as np
import exudyn as exu
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from exudyn.utilities import *
from beams import EulerBernulliBeam, BeamTemplate
from utilities import *
from pathlib import Path
from tqdm import tqdm
print('EXUDYN says hello with version', exu.config.Version() + '.')

#++++++++++++++++++++++++++++++++++++++++++++++++++
# setup of 'test' beams with which the experiments are made
length = 5
beamTemplateRubber = BeamTemplate(physicsLength=length,
                                  physicsAxialStiffness=2e7, physicsBendingStiffness=5e4,
                                  physicsMassPerLength=78, nElements=40)

#++++++++++++++++++++++++++++++++++++++++++++++++++
# some general settings for plotting
textWidth = 5.15
SetMatplotlibSettings()
standardColors, labelList, rawDataLabelList, markerList, timeStr, slidingStr = GetStandardValues()
parentDir = Path.cwd().parent
figPath = parentDir / 'LaTeX' / 'figures'

#%% plot interpolations
PrintHeadlineToConsole(text='Plot Interpolations')
alpha = np.linspace(0, 1, 500)
w = 3*alpha**2 - 2*alpha**3
sigma = (alpha >= 0.5).astype(float)
figHermite, axHermite = plt.subplots(1, 1, figsize=[textWidth, 3])
axHermite.plot(alpha, sigma, '*-', label='no interp.', markevery=20)
axHermite.plot(alpha, w, 'o-', label='Hermite interp.', markevery=20)
axHermite.legend()
axHermite.set_xlabel(r'$\alpha$')
axHermite.set_ylabel(r'$w$')
axHermite.grid()
plt.tight_layout()
SaveFig(figHermite, figPath, 'M_bindingVectorInterpolation', closeFig=True)

#%% plot snapshots for three sliding joint positions for illustration
PrintHeadlineToConsole(text='Plot Snapshots for Illustration')
beamIllustration = EulerBernulliBeam(beamTemplate=beamTemplateRubber) # use this beam for illustration purposes
eigenmodesLiftedAndMassNormalizedListIllustration = beamIllustration.GetGlobalMassNormalizedEigenmodesListForNumberOfConstrainedNodes(nNodesConstrained=3, nEigenmodes=3)
figEigenmodes = beamIllustration.PlotEigenmodesSets(eigenmodesLiftedAndMassNormalizedListIllustration, nEigenmodesToPlot=3,
                                                    nodeNumbersConstrainedList=beamIllustration.nodeNumbersConstrainedList,
                                                    figSize=[textWidth, 4])
SaveFig(figEigenmodes, figPath, 'M_eigenmodesIllustration', closeFig=True)

#%% define and plot a test slidingJoint and loadTip TEST function
tEnd = 15
h = 1e-3 
def SlidingJointTestFunction(t, startPosition=length/4, stopPosition=length*(3/4),
                         startTime=2.5, stopTime=7.5):
    if t < startTime:
        return startPosition
    if t > stopTime:
        return stopPosition
    else:
        return startPosition + ((stopPosition-startPosition)/(stopTime-startTime)) * (t-startTime)
def LoadTipTestFunction(mbs, t, load):
    A = - 50
    fStart = 1      # [Hz] starting frequency of the chirp
    fEnd = 10       # [Hz] last frequency of the chirp
    TChirp = 10     # [s] duration of the chirp
    if t <= TChirp:
        f = fStart + (fEnd - fStart) * t / TChirp
        return A * np.sin(2*np.pi*f*t)
    else:
        return A
    
#++++++++++++++++++++++++++++++++++++++++++++++++++
# plot the sliding joint and load that is used for testing
figTest, axLoad = plt.subplots(1, 1, figsize=[textWidth, 3])
time = np.linspace(0, tEnd, 5000)
blue = '#1f77b4'
axLoad.plot(time, [LoadTipTestFunction(None, t, None) for t in time], color=blue, alpha=0.7)
axLoad.grid()
axLoad.set_xlabel(timeStr)
axLoad.tick_params(axis='y', colors=blue)
axLoad.set_ylabel('tip force in N', color=blue)
markerStyle = 'x'
axSliding = axLoad.twinx()
axSliding.tick_params(axis='y', colors='orange')
axSliding.plot(time, [SlidingJointTestFunction(t) for t in time], color='orange', marker=markerStyle, markevery=300, linewidth=1.3)
axSliding.set_ylabel(f'(x) sliding joint position in m', color='orange')
plt.tight_layout()
SaveFig(figTest, figPath, 'M_testSetup', closeFig=False)

#%% variation experiment to test good number of snapshots, eigenmodes per snapshot, and number of global modes
PrintHeadlineToConsole(text='Variation Experiment')
# set up the parameters for the experiment
loadResults = True
nSnapshotsList = [4, 8, 12, 16]
nEigenmodesPerSnapshotList = [3, 4, 5, 6]
nGlobalModesList = [4, 8, 12, 16, 20, 24, 28, 32]
if not loadResults:
    # precompute info for status bar
    nValidVariations = sum(1 for nSnapshots in nSnapshotsList for nEigenmodesPerSnapshot in nEigenmodesPerSnapshotList for nGlobalModes in nGlobalModesList
                           if nGlobalModes <= (nEigenmodesPerSnapshot * nSnapshots))
    
    #++++++++++++++++++++++++++++++++++++++++++++++++++
    # initialize beam (to inherit some data to ANCF beam) and mbs
    beamVariationPMR = EulerBernulliBeam(beamTemplate=beamTemplateRubber,
                                         slidingJointFunction=SlidingJointTestFunction, loadTipFunction=LoadTipTestFunction,
                                         interpolateBCForSlidingJoint=True)
    SC = exu.SystemContainer()
    mbs = SC.AddSystem()
    
    #++++++++++++++++++++++++++++++++++++++++++++++++++
    # retreive reference solution via ANCF beam with sliding joint
    nGround = mbs.AddNode(NodePointGround())
    # this is needed to constrain ANCF beam on the 'left' side
    mCoordGround = mbs.AddMarker(MarkerNodeCoordinate(nodeNumber=nGround, coordinate=0,
                                                      visualization=VMarkerNodeCoordinates(show=False)))
    sensorOutputVariableTypeVariationANCFList = [exu.OutputVariableType.Displacement] # here, we only need displacement
    sDictVariationANCF = AddANCFBeamWithSlidingJointToMBSAndAddSensors(mbs, SlidingJointTestFunction, LoadTipTestFunction, length=length, rhoA=beamVariationPMR.rhoA, EI=beamVariationPMR.EI, EA=beamVariationPMR.EA,
                                                                       nElementsANCF=10, mCoordGround=mCoordGround, sensorOutputVariableTypeANCFList=sensorOutputVariableTypeVariationANCFList,
                                                                       beamNodeReferencePositionList=beamVariationPMR.nodeReferencePositionsList)
    mbs.Assemble()
    simulationSettings = GetStandardSimulationAndSetVisualizationSettings(tEnd, h, exportFrames=False)
    mbs.SolveDynamic(simulationSettings=simulationSettings)
    outputDictVariationANCF = GetOutputDictANCFFromSDictANCF(mbs, sDictANCF=sDictVariationANCF, sensorOutputVariableTypeANCFList=sensorOutputVariableTypeVariationANCFList)
    displacementVariationANCF = outputDictVariationANCF['OutputVariableType.Displacement']
    
    #++++++++++++++++++++++++++++++++++++++++++++++++++
    # do variation
    variationResults = np.full((len(nSnapshotsList), len(nEigenmodesPerSnapshotList), len(nGlobalModesList)), -1.0)
    with tqdm(total=nValidVariations, desc='Parameter variation', unit='case') as pbar:
        for i, nSnapshots in enumerate(nSnapshotsList):
            for j, nEigenmodesPerSnapshot in enumerate(nEigenmodesPerSnapshotList):
                for k, nGlobalModes in enumerate(nGlobalModesList):
                    if nGlobalModes <= (nEigenmodesPerSnapshot * nSnapshots):
                        
                        #++++++++++++++++++++++++++++++++++++++++++++++++++
                        # set up mbs and beam for this variation
                        mbs.Reset()
                        nGround = mbs.AddNode(NodePointGround())
                        mCoordsGround = mbs.AddMarker(MarkerNodeCoordinates(nodeNumber=nGround, visualization=VMarkerNodeCoordinates(show=False)))
                        beamVariationPMR = EulerBernulliBeam(beamTemplate=beamTemplateRubber,
                                                             slidingJointFunction=SlidingJointTestFunction, loadTipFunction=LoadTipTestFunction,
                                                             interpolateBCForSlidingJoint=True)
                        
                        #++++++++++++++++++++++++++++++++++++++++++++++++++
                        # GLOBALLY reduced beam
                        beamVariationPMR.GlobalReduceBeamWrapper(nGlobalModes=nGlobalModes, nSnapshots=nSnapshots, nEigenmodesPerSnapshot=nEigenmodesPerSnapshot)
                        sCoordinatesVariationPMR, sCoordinates_tVariationPMR = beamVariationPMR.AddBeamToMBSAndAddSensors(mbs, mCoordsGround=mCoordsGround, show=False)
                        
                        #++++++++++++++++++++++++++++++++++++++++++++++++++
                        # simulate dynamically (WYSWYS)
                        mbs.Assemble()
                        simulationSettings = GetStandardSimulationAndSetVisualizationSettings(tEnd, h, exportFrames=False)
                        mbs.SolveDynamic(simulationSettings=simulationSettings)
                        
                        #++++++++++++++++++++++++++++++++++++++++++++++++++
                        # get results and compute Normalized Root Mean Square Error (NRMSE)
                        _, time, qVariationPMR, q_tVariationPMR = beamVariationPMR.GetTimeAndCoordinates(mbs, sCoordinatesVariationPMR, sCoordinates_tVariationPMR)
                        variationResults[i, j, k] = NRMSEVector(qVariationPMR[:, 1::beamVariationPMR.nDOFsPerNode], displacementVariationANCF[:, 1::3])
                        
                        #++++++++++++++++++++++++++++++++++++++++++++++++++
                        # update pbar
                        pbar.set_postfix({'nSnap': nSnapshots, 'nEig': nEigenmodesPerSnapshot, 'nGlob': nGlobalModes})
                        pbar.update(1)
    np.save('variationSolution', variationResults)
    
#++++++++++++++++++++++++++++++++++++++++++++++++++
# do plotting
variationResults = np.load('variationSolution.npy')
validResults = variationResults[variationResults != -1.0]
vmin, vmax = validResults.min(), validResults.max()
cmap = cm.RdYlGn_r
norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
figVariation = plt.figure(figsize=[textWidth, 6])
axVariation = figVariation.add_subplot(111, projection='3d')
for i, nSnap in enumerate(nSnapshotsList):
    for j, nEig in enumerate(nEigenmodesPerSnapshotList):
        for k, nGlob in enumerate(nGlobalModesList):
            val = variationResults[i, j, k]
            if val == -1.0:
                continue
            axVariation.scatter(nSnap, nEig, nGlob, c=[cmap(norm(val))], s=60, alpha=0.8)
axVariation.set_xlabel('num. snapshots')
axVariation.set_ylabel('num. eigenmodes / snapshot')
axVariation.set_zlabel('num. global modes')
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
cb = figVariation.colorbar(sm, ax=axVariation, orientation='horizontal', label='normalized RMSE of beam displ.', pad=0.10, shrink=0.6)
ticks = np.linspace(vmin, vmax, 4)
cb.set_ticks(ticks)
cb.ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
plt.tight_layout
plt.show()
SaveFig(figVariation, figPath, 'R_variation', closeFig=True)

#%% pick one combination and make plot of tip displacement
# nSnapshots = 12             # number of snapshots used for global reduction
# nEigenmodesPerSnapshot = 4  # eigenmodes used per snapshot
# nGlobalModes = 24           # number of global modes in global reduction basis

# #++++++++++++++++++++++++++++++++++++++++++++++++++
# # initialize beam and mbs
# beamVariationPMR = EulerBernulliBeam(beamTemplate=beamTemplateRubber,
#                                      slidingJointFunction=SlidingJointTestFunction, loadTipFunction=LoadTipTestFunction,
#                                      interpolateBCForSlidingJoint=True)
# SC = exu.SystemContainer()
# mbs = SC.AddSystem()













#%%

nCases = 3
computationTimeListANCF = []
computationTimeListNR = []
computationTimeListPMR = []
nSnapshots = 12
nEigenmodesPerSnapshot = 4
nGlobalModes = 24
SC = exu.SystemContainer()
mbs = SC.AddSystem()
beamComputationNR = EulerBernulliBeam(beamTemplate=beamTemplateRubber,
                                      slidingJointFunction=SlidingJointTestFunction, loadTipFunction=LoadTipTestFunction,
                                      interpolateBCForSlidingJoint=True)
for i in range(nCases):
    mbs.Reset()
    nGround = mbs.AddNode(NodePointGround())
    mCoordGround = mbs.AddMarker(MarkerNodeCoordinate(nodeNumber=nGround, coordinate=0,
                                                      visualization=VMarkerNodeCoordinates(show=False)))
    sensorOutputVariableTypeVariationANCFList = [exu.OutputVariableType.Displacement] # here, we only need displacement
    sDictVariationANCF = AddANCFBeamWithSlidingJointToMBSAndAddSensors(mbs, SlidingJointTestFunction, LoadTipTestFunction, length=length, rhoA=beamComputationNR.rhoA, EI=beamComputationNR.EI, EA=beamComputationNR.EA,
                                                                       nElementsANCF=10, mCoordGround=mCoordGround, sensorOutputVariableTypeANCFList=sensorOutputVariableTypeVariationANCFList,
                                                                       beamNodeReferencePositionList=beamComputationNR.nodeReferencePositionsList)
    mbs.Assemble()
    simulationSettings = GetStandardSimulationAndSetVisualizationSettings(tEnd, h, exportFrames=False)
    startTime = tm.time()
    mbs.SolveDynamic(simulationSettings=simulationSettings)
    duration = tm.time() - startTime
    computationTimeListANCF += [duration]
for i in range(nCases):
    mbs.Reset()
    nGround = mbs.AddNode(NodePointGround())
    mCoordsGround = mbs.AddMarker(MarkerNodeCoordinates(nodeNumber=nGround, visualization=VMarkerNodeCoordinates(show=False)))
    beamComputationNR = EulerBernulliBeam(beamTemplate=beamTemplateRubber,
                                          slidingJointFunction=SlidingJointTestFunction, loadTipFunction=LoadTipTestFunction,
                                          interpolateBCForSlidingJoint=True)
    _, _ = beamComputationNR.AddBeamToMBSAndAddSensors(mbs, mCoordsGround=mCoordsGround, show=False)
    
    #++++++++++++++++++++++++++++++++++++++++++++++++++
    # simulate dynamically (WYSWYS)
    mbs.Assemble()
    simulationSettings = GetStandardSimulationAndSetVisualizationSettings(tEnd, h, exportFrames=False)
    # supress warnings here
    startTime = tm.time()
    mbs.SolveDynamic(simulationSettings=simulationSettings)
    duration = tm.time() - startTime
    computationTimeListNR += [duration]
for i in range(nCases):
    mbs.Reset()
    nGround = mbs.AddNode(NodePointGround())
    mCoordsGround = mbs.AddMarker(MarkerNodeCoordinates(nodeNumber=nGround, visualization=VMarkerNodeCoordinates(show=False)))
    beamComputationPMR = EulerBernulliBeam(beamTemplate=beamTemplateRubber,
                                           slidingJointFunction=SlidingJointTestFunction, loadTipFunction=LoadTipTestFunction,
                                           interpolateBCForSlidingJoint=True)
    
    #++++++++++++++++++++++++++++++++++++++++++++++++++
    # GLOBALLY reduced beam
    beamComputationPMR.GlobalReduceBeamWrapper(nGlobalModes=nGlobalModes, nSnapshots=nSnapshots, nEigenmodesPerSnapshot=nEigenmodesPerSnapshot)
    _, _ = beamComputationPMR.AddBeamToMBSAndAddSensors(mbs, mCoordsGround=mCoordsGround, show=False)
    
    #++++++++++++++++++++++++++++++++++++++++++++++++++
    # simulate dynamically (WYSWYS)
    mbs.Assemble()
    simulationSettings = GetStandardSimulationAndSetVisualizationSettings(tEnd, h, exportFrames=False)
    startTime = tm.time()
    mbs.SolveDynamic(simulationSettings=simulationSettings)
    duration = tm.time() - startTime
    computationTimeListPMR += [duration]

#%%
# Calculate means and standard deviations
means = [np.mean(computationTimeListANCF), np.mean(computationTimeListNR), np.mean(computationTimeListPMR)]
# stds = [np.mean(computationTimeListANCF), np.std(computationTimeListNR), np.std(computationTimeListPMR)]
# Bar positions
x_pos = np.arange(len(means))
# Create bar chart with error bars
plt.bar(x_pos, means, capsize=10, alpha=0.7, color=[standardColors[0], standardColors[1], standardColors[2]])
plt.xticks(x_pos, ['ANCF', 'non-reduced', 'globally reduced'])
plt.ylabel('mean computation time in s')
plt.show()
plt.grid()