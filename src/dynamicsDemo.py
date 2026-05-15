#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#
# Details:  This is a demo script for the 'BoomPMR'-Project.
#           Run it to test specific settings in a fast and self-contained
#           manner. The script uses the Absolute Nodal Coordinate Formulation
#           (ANCF) with a sliding joint as the reference formulation and
#           compares a Non-Reduced (NR) Euler-Bernulli (EB) beam, OPTIONALLY a
#           Specifically for one configuration Reduced (SR) EB beam (which
#           seems to break the solver) and a globally (Parametric Model order
#           Reduction (PMR)) reduced EB beam with it.
#           The sliding joint for the EB beams is implemented using Lagrange
#           multipliers (q-derivative of constraint equations -> Jacobian C_q).
#           Remarks on ghraphics: The beam class ('beams.py') has it's own
#           graphics functionality for the beam but not constraints etc.. Here,
#           'utility.py' provides functionality.
# Author:   Sebastian Weyrer
# Date:     2026-05-15
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import numpy as np
import exudyn as exu
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from exudyn.utilities import *
from beams import EulerBernulliBeam, BeamTemplate
from utilities import *
PrintHeadlineToConsole(text='BoomPMR')
print('EXUDYN says hello with version', exu.config.Version() + '.')
plt.close('all')

#++++++++++++++++++++++++++++++++++++++++++++++++++
# define bahvior of script and the boom
# parameters of the boom
length = 5                                  # [m] length of boom
rhoA = 78 #490 #78                          # [kg/m] mass per length of beam
EA = 2e7                                    # [N] axial stiffness of beam
EI = 5e4 #6.8e7 # 5e4 #6.8e7#               # [Nm^2] bending stiffness of beam
nElements = 50                              # number of elements (discretization) the beam (NR and PMR) is made out of
nElementsANCF = 20                          # number of elements (discretization) the beam (ANCF) is made out of    
nSnapshots = 5                              # number of snapshots used for global reduction
nEigenmodesPerSnapshot = 6                  # eigenmodes used per snapshot
nGlobalModes = 20                           # number of global modes in global reduction basis
# define behavior of script
tEnd = 15                                   # [s] duration of the experiment
h = 1e-3                                    # [s] step size of integration
addSpecificallyReducedBeam = False          # add SR beam (which breaks the solver)
useRenderer = True                          # use EXUDYN's renderer?
exportFrames = False                        # export frames
solveInMinimalCoordinatesWithRadau = False  # also solve in minimal coordinates (for the last sliding position) using radau (to check coordinatesVectorConstraint implementation)
# define drawing properties
width = 0.25                                                            # [m] beam width for drawing
yReferenceOffsetANCF = 1.5 if addSpecificallyReducedBeam else 1         # y-offset of ANCF beam (to separate them for better view)
yReferenceOffsetBeamNR = 0.5 if addSpecificallyReducedBeam else 0       # y-offset of NR beam (to separate them for better view)
yReferenceOffsetBeamSR = -0.5                                           # y-offset of SR beam (to separate them for better view)
yReferenceOffsetBeamPMR = -1.5 if addSpecificallyReducedBeam else -1    # y-offset of PMR beam (to separate them for better view)
xTextOffset = 0.1                                                       # x-offset of the text
yTextOffset = 0.3                                                       # y-offset of the text (measured to the y-offset of the beam)
zOffsetForDrawing = 0.01                                                # to see the constraints in renderer, we add a small offset in z-direction
# define parameters of sliding joint movement
tExperiment = tEnd  # the experiment duration is equal the simulation duration
startPosition = length/4
stopPosition = length*(3/4)
startTime = tExperiment/4
stopTime = tExperiment*(3/4)
A = 50 # max. amplitude of load functions
# special settings for outputs (sensor outputs) of ANCF simulation
sensorOutputVariableTypeANCFList = [exu.OutputVariableType.Displacement,
                                    exu.OutputVariableType.Rotation,
                                    exu.OutputVariableType.Velocity,
                                    exu.OutputVariableType.AngularVelocity,
                                    exu.OutputVariableType.StrainLocal,
                                    exu.OutputVariableType.CurvatureLocal,
                                    exu.OutputVariableType.ForceLocal,
                                    exu.OutputVariableType.TorqueLocal]
# standard values for visualization, plotting, etc. (standardColors is used as color for EXUDYN visualization of beam)
SetMatplotlibSettings()
standardColors, labelList, rawDataLabelList, markerList, timeStr, slidingStr = GetStandardValues(nGlobalModes=nGlobalModes)

#%% define a function that returns a sliding joint position and a load on the tip
print('Info: Velocity of sliding joint:', str(abs(stopPosition-startPosition)/(stopTime-startTime)) + 'm/s.')
# def SlidingJointFunction(t, startPosition=startPosition, stopPosition=stopPosition,
#                          startTime=startTime, stopTime=stopTime):
#     if solveInMinimalCoordinatesWithRadau:
#         return length/2
#     if t < startTime:
#         return startPosition
#     if t > stopTime:
#         return stopPosition
#     else:
#         return startPosition + ((stopPosition-startPosition)/(stopTime-startTime)) * (t-startTime)
# NOTE: this only works for interpolated BC since with no interpolation node-flickering is prevented
def SlidingJointFunction(t, startPosition=startPosition, stopPosition=stopPosition,
                         startTime=startTime, stopTime=stopTime):
    if t < startTime:
        return startPosition
    if t > stopTime:
        return startPosition
    else:
        duration = (stopTime-startTime) / 2
        if t > startTime and t < (startTime+duration):
            return startPosition + ((stopPosition-startPosition)/(duration)) * (t-startTime) 
        else:
            return stopPosition - ((stopPosition-startPosition)/(duration)) * (t-(startTime+duration))
# def LoadTipFunction(mbs, t, load):
#     # # excite the 1st eigenmode of sliding joint being at node 22
#     # omega = np.sqrt(111.31172747)
#     # # excite the 2nd eigenmode of sliding joint being at node 22
#     # omega = np.sqrt(1747.41342898)
#     # # excite the 3rd eigenmode of sliding joint being at node 22
#     # omega = np.sqrt(8417.44841418)
#     # return 50 * sin(omega*t)
#     return -A
# def LoadTipFunction(mbs, t, load):
#     A = - 50
#     fStart = 1      # [Hz] starting frequency of the chirp
#     fEnd = 10       # [Hz] last frequency of the chirp
#     TChirp = 10     # [s] duration of the chirp
#     if t <= TChirp:
#         f = fStart + (fEnd - fStart) * t / TChirp
#         return A * np.sin(2*np.pi*f*t)
#     else:
#         return A
def LoadTipFunction(mbs, t, load):
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

#%% create the beams (non reduced and globally reduced)
PrintHeadlineToConsole(text='Setting up beams')
beamTemplateDemo = BeamTemplate(physicsLength=length, physicsAxialStiffness=EA, physicsBendingStiffness=EI,
                                physicsMassPerLength=rhoA, nElements=nElements, widthForDrawing=width,
                                slidingJointFunction=SlidingJointFunction, loadTipFunction=LoadTipFunction)
beamNR = EulerBernulliBeam(beamTemplate=beamTemplateDemo, color=standardColors[1], positionReference=[0, yReferenceOffsetBeamNR, 0])
if addSpecificallyReducedBeam:
    beamSR = EulerBernulliBeam(beamTemplate=beamTemplateDemo, positionReference=[0, yReferenceOffsetBeamSR, 0])
beamPMR = EulerBernulliBeam(beamTemplate=beamTemplateDemo, color=standardColors[2], positionReference=[0, yReferenceOffsetBeamPMR, 0])

#%% do SPECIFIC reduction
if addSpecificallyReducedBeam:
    beamSR.SetConstrainedNodeNumberDueToSlidingJoint(np.zeros(beamSR.nDOFs), 0) # set to one specific BC (first one)
    eigenmodesLiftedAndMassNormalizedListSR = beamSR.GetGlobalMassNormalizedEigenmodesListForNumberOfConstrainedNodes(nodeNumbersConstrainedList=[beamSR.nodeNumberConstrainedCurrent],
                                                                                                                      nEigenmodes=nEigenmodesPerSnapshot)
    beamSR.PlotEigenmodesSets(eigenmodesList=eigenmodesLiftedAndMassNormalizedListSR,
                              nodeNumbersConstrainedList=beamSR.nodeNumbersConstrainedList, figSize=[5, 2])
    beamSR.ReduceBeam(reductionBasis=eigenmodesLiftedAndMassNormalizedListSR[0])

#%% do GLOBAL reduction
beamPMR.GlobalReduceBeamWrapper(nGlobalModes=nGlobalModes, nSnapshots=nSnapshots, nEigenmodesPerSnapshot=nEigenmodesPerSnapshot,
                                plot=True, figSize=[5, 7], verbose=True)

#%% set up simulations for ANCF beam, non reduced, (specifically rediced), and globally reduced beam
SC = exu.SystemContainer()
mbs = SC.AddSystem()

#++++++++++++++++++++++++++++++++++++++++++++++++++
# add ground and show the constraints for all three beams
def GroundGraphicsDataUserFunction(mbs, objectNumber):
    return GetGraphicsOfConstraintsAndAddBasis(mbs, objectNumber, SlidingJointFunction,
                                               zOffsetForDrawing=zOffsetForDrawing, yReferenceOffsetANCF=yReferenceOffsetANCF, yReferenceOffsetBeamNR=yReferenceOffsetBeamNR, yReferenceOffsetBeamPMR=yReferenceOffsetBeamPMR,
                                               xTextOffset=xTextOffset, yTextOffset=yTextOffset)
oGround = mbs.CreateGround(graphicsDataUserFunction=GroundGraphicsDataUserFunction)
nGround = mbs.AddNode(NodePointGround())
mCoordsGround = mbs.AddMarker(MarkerNodeCoordinates(nodeNumber=nGround, visualization=VMarkerNodeCoordinates(show=False)))
# this is needed to constrain ANCF beam on the 'left' side
mCoordGround = mbs.AddMarker(MarkerNodeCoordinate(nodeNumber=nGround, coordinate=0,
                                                  visualization=VMarkerNodeCoordinates(show=False)))

#++++++++++++++++++++++++++++++++++++++++++++++++++
# ANCF beam
sDictANCF = AddANCFBeamWithSlidingJointToMBSAndAddSensors(mbs, SlidingJointFunction, LoadTipFunction, length=length, rhoA=rhoA, EI=EI, EA=EA,
                                                          nElementsANCF=nElementsANCF, yReferenceOffsetANCF=yReferenceOffsetANCF, width=width, show=True,
                                                          mCoordGround=mCoordGround, sensorOutputVariableTypeANCFList=sensorOutputVariableTypeANCFList,
                                                          beamNodeReferencePositionList=beamNR.nodeReferencePositionsList)

#++++++++++++++++++++++++++++++++++++++++++++++++++
# non reduced beam
sCoordinatesNR, sCoordinates_tNR = beamNR.AddBeamToMBSAndAddSensors(mbs, mCoordsGround=mCoordsGround, show=True)

#++++++++++++++++++++++++++++++++++++++++++++++++++
# SPECIFICALLY reduced beam
if addSpecificallyReducedBeam:
    sCoordinatesSR, sCoordinates_tSR = beamSR.AddBeamToMBSAndAddSensors(mbs, mCoordsGround=mCoordsGround, show=True)

#++++++++++++++++++++++++++++++++++++++++++++++++++
# GLOBALLY reduced beam
sCoordinatesPMR, sCoordinates_tPMR = beamPMR.AddBeamToMBSAndAddSensors(mbs, mCoordsGround=mCoordsGround, show=True)

#++++++++++++++++++++++++++++++++++++++++++++++++++
mbs.Assemble()
# mbs.DrawSystemGraph()

#%% simulate dynamically (WYSWYS)
PrintHeadlineToConsole(text='Starting simulations')
simulationSettings = GetStandardSimulationAndSetVisualizationSettings(tEnd, h, exportFrames=exportFrames)
if useRenderer:
    SC.renderer.Start()
    SC.renderer.DoIdleTasks()
mbs.SolveDynamic(simulationSettings=simulationSettings)
if useRenderer:
    SC.renderer.DoIdleTasks()
    SC.renderer.Stop() #safely close rendering window!

#%% solve minimal coordinates using radau to check implementation of constraints
# NOTE: this is always done for the LAST position the sliding joint is at (acc. to SlidingJointFunction)
# so to validate algebraic equations, use constant sliding position at best
if solveInMinimalCoordinatesWithRadau:
    print('Startet integration using Radau ...')
    # cut out the constrained DOFs of mass and stiffness matrices
    DOFsConstrained = (beamNR.nodeNumberConstrainedCurrent * beamNR.nDOFsPerNode) + 1
    indicesToRemove = [0, 1, DOFsConstrained]
    MUnconstrained = np.delete(beamNR.M, indicesToRemove, axis=0)
    MUnconstrained = np.delete(MUnconstrained, indicesToRemove, axis=1)
    MUnconstrainedInv = np.linalg.inv(MUnconstrained)
    KUnconstrained = np.delete(beamNR.K, indicesToRemove, axis=0)
    KUnconstrained = np.delete(KUnconstrained, indicesToRemove, axis=1)
    nDOFsUnconstrained = beamNR.nDOFs - 3 # always three (can be hard-coded here)
    # define system of ODE1
    def SystemOfODE1(t, x, A, B):
        f = np.zeros(nDOFsUnconstrained)
        f[-2] = LoadTipFunction(None, t, None) # apply in y at last node
        dx = A @ x + B @ f
        return dx
    # assemble A and B matrices to have first order ODE system (system is 2*numberOfReducedCoordinates big)
    A = np.block([[np.zeros([nDOFsUnconstrained, nDOFsUnconstrained]), np.eye(nDOFsUnconstrained)],
                  [-MUnconstrainedInv @ KUnconstrained, np.zeros([nDOFsUnconstrained, nDOFsUnconstrained])]])
    B = np.block([[np.zeros([nDOFsUnconstrained, nDOFsUnconstrained])],
                  [MUnconstrainedInv]])
    sensorWritePeriod = simulationSettings.solutionSettings.sensorsWritePeriod
    tEval = np.arange(0, tEnd+sensorWritePeriod, sensorWritePeriod)
    # dynamically simulate the system
    s0 = np.zeros(2*nDOFsUnconstrained)
    sol = solve_ivp(SystemOfODE1,
                    (tEval[0], tEval[-1]),
                    s0,
                    args=(A, B),
                    t_eval=tEval,
                    method='Radau')

#%% get ALL the needed RAW data for plotting
PrintHeadlineToConsole(text='Post Processing')

#++++++++++++++++++++++++++++++++++++++++++++++++++
# ANCF
outputDictANCF = GetOutputDictANCFFromSDictANCF(mbs, sDictANCF=sDictANCF, sensorOutputVariableTypeANCFList=sensorOutputVariableTypeANCFList, verbose=True)
     
#++++++++++++++++++++++++++++++++++++++++++++++++++
# NR
nTimeSteps, time, qNR, q_tNR = beamNR.GetTimeAndCoordinates(mbs, sCoordinatesNR, sCoordinates_tNR)

#++++++++++++++++++++++++++++++++++++++++++++++++++
# PMR
_, _, qPMR, q_tPMR = beamPMR.GetTimeAndCoordinates(mbs, sCoordinatesPMR, sCoordinates_tPMR)

#++++++++++++++++++++++++++++++++++++++++++++++++++
# NR in minimal coordinates
if solveInMinimalCoordinatesWithRadau:
    tipDisplacementNRMC = sol.y[-2, :] # only y-displacement available here

#%% plot the raw data to check consistency
figCoordinatesOverTime, axCoordinatesOverTime = plt.subplots(3, 3, figsize=[16, 8], sharex=True)
# ANCF
axCoordinatesOverTime[0, 0].plot(time, outputDictANCF['OutputVariableType.Displacement'][:, 0::3]) # only x
axCoordinatesOverTime[1, 0].plot(time, outputDictANCF['OutputVariableType.Displacement'][:, 1::3]) # only y
axCoordinatesOverTime[2, 0].plot(time, outputDictANCF['OutputVariableType.Rotation'][:])
# NR
axCoordinatesOverTime[0, 1].plot(time, qNR[:, 0::beamNR.nDOFsPerNode]) # only x
axCoordinatesOverTime[1, 1].plot(time, qNR[:, 1::beamNR.nDOFsPerNode]) # only y
axCoordinatesOverTime[2, 1].plot(time, qNR[:, 2::beamNR.nDOFsPerNode])
# PMR
axCoordinatesOverTime[0, 2].plot(time, qPMR[:, 0::beamPMR.nDOFsPerNode]) # only x
axCoordinatesOverTime[1, 2].plot(time, qPMR[:, 1::beamPMR.nDOFsPerNode]) # only y
axCoordinatesOverTime[2, 2].plot(time, qPMR[:, 2::beamPMR.nDOFsPerNode])
for ax in axCoordinatesOverTime.flat:
    ax.grid()
for ax, title in zip(axCoordinatesOverTime[0], labelList):
    ax.set_title(title)
for ax, label in zip(axCoordinatesOverTime[:, 0], rawDataLabelList):
    ax.set_ylabel(label)
for ax in axCoordinatesOverTime[-1, :]:
    ax.set_xlabel(timeStr)
plt.tight_layout()

#%% plot the y-displacement of the tip
markerInterval = 30
figTipDisplacementOverTime, axTipDisplacementOverTime = plt.subplots(1, 1, figsize=[8, 4])
markEvery = (0, markerInterval)
axTipDisplacementOverTime.plot(time, outputDictANCF['OutputVariableType.Displacement'][:, -2],
                               markerList[0], label=labelList[0], markevery=markEvery,
                               color=standardColors[0])
markEvery = (int(markerInterval/3), markerInterval)
axTipDisplacementOverTime.plot(time, qNR[:, -2],
                               markerList[1], label=labelList[1], markevery=markEvery,
                               color=standardColors[1])
markEvery = (int(2*markerInterval/3), markerInterval)
axTipDisplacementOverTime.plot(time, qPMR[:, -2],
                               markerList[2], label=labelList[2], markevery=markEvery,
                               color=standardColors[2])
if solveInMinimalCoordinatesWithRadau:
    axTipDisplacementOverTime.plot(time, tipDisplacementNRMC,
                                   markerList[3], label=labelList[3], markevery=markEvery,
                                   color=standardColors[3])
axTipDisplacementOverTime.set_xlabel(timeStr)
axTipDisplacementOverTime.set_ylabel('tip displacement in m')
axTipDisplacementOverTime.axvspan(startTime, stopTime, color='gray', alpha=0.2, label=slidingStr)
axTipDisplacementOverTime.legend()
axTipDisplacementOverTime.grid()
plt.tight_layout()

#%% plot error of tip displacement over time
# compute absolute error
absErrorTipDisplacementNR = np.abs(qNR[:, -2] - outputDictANCF['OutputVariableType.Displacement'][:, -2])
absErrorTipDisplacementPMR = np.abs(qPMR[:, -2] - outputDictANCF['OutputVariableType.Displacement'][:, -2])
if solveInMinimalCoordinatesWithRadau:
    absErrorTipDisplacementNRMC = np.abs(tipDisplacementNRMC - outputDictANCF['OutputVariableType.Displacement'][:, -2])
    figTipDisplacementErrorMCOverTime, axTipDisplacementErrorMCOverTime = plt.subplots(1, 1, figsize=[8, 4])
    axTipDisplacementErrorMCOverTime.plot(time, absErrorTipDisplacementNRMC,
                                          markerList[4], label=labelList[4], markevery=markEvery,
                                          color=standardColors[4])
# compute SMA
SMAWindow = 50
timeSMAError, absErrorTipDisplacementNRSMA = SMA(time, absErrorTipDisplacementNR, SMAWindow, 'valid')
_, absErrorTipDisplacementPMRSMA = SMA(time, absErrorTipDisplacementPMR, SMAWindow, 'valid')

#++++++++++++++++++++++++++++++++++++++++++++++++++
# plot
markEvery = 20
figErrorTipDisplacementOverTime, axErrorTipDisplacementOverTime = plt.subplots(1, 1, figsize=[8, 4])
axErrorTipDisplacementOverTime.plot(time, absErrorTipDisplacementNR, alpha=0.2, color=standardColors[1])
axErrorTipDisplacementOverTime.plot(time, absErrorTipDisplacementPMR, alpha=0.2, color=standardColors[2])
axErrorTipDisplacementOverTime.plot(timeSMAError, absErrorTipDisplacementNRSMA,
                                    markerList[1], label=labelList[1], markevery=markEvery,
                                    color=standardColors[1])
axErrorTipDisplacementOverTime.plot(timeSMAError, absErrorTipDisplacementPMRSMA,
                                    markerList[2], label=labelList[2], markevery=markEvery,
                                    color=standardColors[2])
axErrorTipDisplacementOverTime.set_xlabel(timeStr)
axErrorTipDisplacementOverTime.set_ylabel('abs. error (SMA) tip displ. against ANCF in m ')
axErrorTipDisplacementOverTime.axvspan(startTime, stopTime, color='gray', alpha=0.2, label=slidingStr)
axErrorTipDisplacementOverTime.legend()
axErrorTipDisplacementOverTime.grid()
plt.tight_layout()

#%% compute one scalar error value
NRMSENR = NRMSE(qNR[:, -2], outputDictANCF['OutputVariableType.Displacement'][:, -2])
NRMSEPMR = NRMSE(qPMR[:, -2], outputDictANCF['OutputVariableType.Displacement'][:, -2])
print('Non-reduced NRMS tip-displacement error against ANCF:', str(round(NRMSENR*100, 2)) + '%.')
print('Globally reduced NRMS tip-displacement error:', str(round(NRMSEPMR*100, 2)) + '%.')
NRMSEVectorAgainstANCF = NRMSEVector(qNR[:, 1::beamNR.nDOFsPerNode], outputDictANCF['OutputVariableType.Displacement'][:, 1::3])
NRMSEVectorAgainstNR = NRMSEVector(qPMR[:, 1::beamPMR.nDOFsPerNode], qNR[:, 1::beamNR.nDOFsPerNode])
print('Non-reduced NRMS beam-displacement error against ANCF:', str(round(NRMSEVectorAgainstANCF*100, 2)) + '%.')
print('Globally reduced NRMS whole beam error against NR:', str(round(NRMSEVectorAgainstNR*100, 2)) + '%.')

#%% plot the energies over time
# compute the energies
# ANCF
EANCF = GetKineticAndStrainAndPotentialANCFEnergyFromOutputDictANCF(outputDictANCF, nTimeSteps=nTimeSteps, rhoA=rhoA, LoadTipFunction=LoadTipFunction,
                                                                    length=length, nBeamElements=nElements, sensorsWritePeriod=simulationSettings.solutionSettings.sensorsWritePeriod)
# NR
ENR = beamNR.GetKineticAndStrainAndPotentialBeamEnergy(qNR, q_tNR, sensorWritePeriod=simulationSettings.solutionSettings.sensorsWritePeriod)
# PMR
EPMR = beamPMR.GetKineticAndStrainAndPotentialBeamEnergy(qPMR, q_tPMR, sensorWritePeriod=simulationSettings.solutionSettings.sensorsWritePeriod)

#++++++++++++++++++++++++++++++++++++++++++++++++++
# plot
markEvery = 20
figEnergyOverTime, axEnergyOverTime = plt.subplots(1, 1, figsize=[8, 4])
axEnergyOverTime.plot(time, EANCF, markerList[0], label=labelList[0], markevery=markEvery, color=standardColors[0])
axEnergyOverTime.plot(time, ENR, markerList[1], label=labelList[1], markevery=markEvery, color=standardColors[1])
axEnergyOverTime.plot(time, EPMR, markerList[2], label=labelList[2], markevery=markEvery, color=standardColors[2])
axEnergyOverTime.set_xlabel(timeStr)
axEnergyOverTime.set_ylabel('Hamiltonian of the system in J')
# axEnergyOverTime.set_yscale('log')
axEnergyOverTime.axvspan(startTime, stopTime, color='gray', alpha=0.2, label=slidingStr)
axEnergyOverTime.legend()
axEnergyOverTime.grid()
plt.tight_layout()
