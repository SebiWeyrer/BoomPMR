#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#
# Details:  Two 2D booms, formulated using ANCF, are simulated, sliding on each
#           other. Boom0 is fixed to ground at it's ends, boom1 slides along
#           boom0, such that a reference solution for changing BCs is gained.
#           Note that 'S' always describes the Start of the boom and 'E'
#           refers to the End of the boom (boom0 is orientated along the
#           pos. x-axis and boom1 is oriented along the neg. x-axis).
# Authors:  Sebastian Weyrer
# Date:     2025-12-16
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import numpy as np
import exudyn as exu
from exudyn.utilities import *
import exudyn.graphics as graphics
print('EXUDYN says hello with version', exu.config.Version())

#++++++++++++++++++++++++++++++++++++++++++++++++++
# define bahvior of script and the boom
# parameters of the boom
length = 5                  # [m] length of the booms
height = 0.25               # [m] height of boom1 (not used in EOMs, boom0 is drawn automatically higher)
rhoA = 7                    # [kg/m] mass per length of the booms
EA = 1000                   # [N] axial stiffness of boom1 (axial and bending stiffness (/m) of boom0)
EI = 500                    # [Nm^2] bending stiffness of boom1
dA = 0                      # [N/s] axial damping of booms
dI = EI*0                   # [Nm^2/s] bending damping of booms 
nElements = 5               # number of elements the booms are made out of
loadTip = -5                # [N] load at the tip of the boom
transitionStayTime = 20     # [s] time the boom should stay at a specified position
transitionTime = 6          # [s] time of a transition from one extraction state x to another
# the following defines whether boom0 should be supported from two sides or only fixed at start (if False)
constrainBothSidesOfBoom0 = True
xList = [(i+1) * (length / nElements) for i in range(4)]
# behavior of the script
useSolutionViewer = True 
useRenderer = True
# automatically computed parameters
nX = len(xList)
segmentTime = transitionStayTime+transitionTime
totalExperimentTime = nX * segmentTime - transitionTime
elementLength = length / nElements
print('Length of each boom element: ' +  str(elementLength) + 'm')
print('Duration of experiment: ' +  str(totalExperimentTime) + 's')
print('Extraction length in experiment x=', xList)

#%% set up the multibody system
SC = exu.SystemContainer()
mbs = SC.AddSystem()

#++++++++++++++++++++++++++++++++++++++++++++++++++
# add a ground object and make coordinate markers to constrain boom0
graphicsDataGround = []
graphicsDataGround += [graphics.Frame(length=1)]
oGround = mbs.AddObject(ObjectGround(visualization=VObjectGround(graphicsData=graphicsDataGround)))
nGroundS = mbs.AddNode(NodePointGround(referenceCoordinates=[-elementLength, 0, 0])) # start of boom0
nGroundE = mbs.AddNode(NodePointGround(referenceCoordinates=[elementLength*(nElements-1), 0, 0])) # end of boom0
# for NodePintGround we only need one coordinate each to constrain
mNCGroundS = mbs.AddMarker(MarkerNodeCoordinate(nodeNumber=nGroundS, coordinate=0))
mNCGroundE = mbs.AddMarker(MarkerNodeCoordinate(nodeNumber=nGroundE, coordinate=0))

#++++++++++++++++++++++++++++++++++++++++++++++++++
# make boom 0
# set up NodePoint2DSlope1 nodes for boom0, and make it as ANCF elements
# coordinates of node: x, y, slope vector x-coordinate, slope vector y-coordinate
# start and end of boom0 are defined acc. to positive x-axis
nEleBoom0S = mbs.AddNode(NodePoint2DSlope1(referenceCoordinates=[-elementLength, 0, 1, 0],
                                          initialCoordinates=[0]*4,
                                          initialVelocities=[0]*4))
if constrainBothSidesOfBoom0:
    # now add coordinate markers to the first node of boom0 as it can be constrained to ground here
    mNCBoom0SX = mbs.AddMarker(MarkerNodeCoordinate(nodeNumber=nEleBoom0S, coordinate=0))
    mNCBoom0SY = mbs.AddMarker(MarkerNodeCoordinate(nodeNumber=nEleBoom0S, coordinate=1))
mBBoom0List = [] # list of body markers to boom0
oBoom0List = [] # list of objects making boom0
print('=====================================================')
print('Adding boom0 ...')
for i in range(nElements):
    print('End of element', (i+1), 'of boom0:', elementLength*i)
    nEleBoom0E = mbs.AddNode(NodePoint2DSlope1(referenceCoordinates=[elementLength*i, 0, 1, 0],
                                               initialCoordinates=[0]*4,
                                               initialVelocities=[0]*4))
    oBoom0List += [mbs.AddObject(ObjectANCFCable2D(nodeNumbers=[nEleBoom0S, nEleBoom0E],
                                                   physicsLength=elementLength,
                                                   physicsMassPerLength=rhoA,
                                                   physicsBendingStiffness=EA*100, # set this very high as well
                                                   physicsAxialStiffness=EA*100,
                                                   physicsBendingDamping=dI,
                                                   physicsAxialDamping=dA,
                                                   visualization=VObjectALEANCFCable2D(drawHeight=height*1.2, 
                                                                                       color=color4dodgerblue)))]
    mBBoom0List += [mbs.AddMarker(MarkerBodyCable2DCoordinates(bodyNumber=oBoom0List[-1]))]
    nEleBoom0S = nEleBoom0E
print('Boom0 has marker numbers:', mBBoom0List)
# now add position marker to the last node of boom0 as here the fixed part of a slidingJoint will be
mNBoom0E = mbs.AddMarker(MarkerNodePosition(nodeNumber=nEleBoom0E))
if constrainBothSidesOfBoom0:
    # now add coordinate markers to the last node of boom0 as it can be constrained to ground here
    mNCBoom0EX = mbs.AddMarker(MarkerNodeCoordinate(nodeNumber=nEleBoom0E, coordinate=0))
    mNCBoom0EY = mbs.AddMarker(MarkerNodeCoordinate(nodeNumber=nEleBoom0E, coordinate=1))

#++++++++++++++++++++++++++++++++++++++++++++++++++
# make boom 1
# set up NodePoint2DSlope1 nodes for boom0, and make it as ANCF elements
# coordinates of node: x, y, slope vector x-coordinate, slope vector y-coordinate
# start and end of boom1 are defined acc. to negative x-axis
nEleBoom1S = mbs.AddNode(NodePoint2DSlope1(referenceCoordinates=[length, 0, -1, 0],
                                           initialCoordinates=[0]*4,
                                           initialVelocities=[0]*4))
# now add position marker to the last node of boom1 as here load will be applied
mNBoom1S = mbs.AddMarker(MarkerNodePosition(nodeNumber=nEleBoom1S))
mBBoom1List = [] # list of body markers to boom1
oBoom1List = [] # list of objects making boom1
print('=====================================================')
print('Adding boom1 ...')
for i in range(nElements):
    print('End of element', (i+1), 'of boom1:', length-(elementLength*(i+1)))
    nEleBoom1E = mbs.AddNode(NodePoint2DSlope1(referenceCoordinates=[length-(elementLength*(i+1)), 0, -1, 0],
                                               initialCoordinates=[0]*4,
                                               initialVelocities=[0]*4))
    oBoom1List += [mbs.AddObject(ObjectANCFCable2D(nodeNumbers=[nEleBoom1S, nEleBoom1E],
                                                   physicsLength=elementLength,
                                                   physicsMassPerLength=rhoA,
                                                   physicsBendingStiffness=EI,
                                                   physicsAxialStiffness=EA,
                                                   physicsBendingDamping=dI,
                                                   physicsAxialDamping=dA,
                                                   visualization=VObjectALEANCFCable2D(drawHeight=height, 
                                                                                       color=color4grey)))]
    mBBoom1List += [mbs.AddMarker(MarkerBodyCable2DCoordinates(bodyNumber=oBoom1List[-1]))]
    nEleBoom1S = nEleBoom1E
print('Boom0 has marker numbers:', mBBoom1List)
# now add position marker to the last node of boom1 as here the fixed part of a slidingJoint will be
mNBoom1E = mbs.AddMarker(MarkerNodePosition(nodeNumber=nEleBoom1E))

#++++++++++++++++++++++++++++++++++++++++++++++++++
# fix boom0 depending on mode
if constrainBothSidesOfBoom0:
    oCCBoom0SX = mbs.AddObject(CoordinateConstraint(markerNumbers=[mNCGroundS, mNCBoom0SX]))
    oCCBoom0SY = mbs.AddObject(CoordinateConstraint(markerNumbers=[mNCGroundS, mNCBoom0SY]))
    oCCBoom0EX = mbs.AddObject(CoordinateConstraint(markerNumbers=[mNCGroundE, mNCBoom0EX]))
    oCCBoom0EX = mbs.AddObject(CoordinateConstraint(markerNumbers=[mNCGroundE, mNCBoom0EY]))
else:
    mbs.CreateCoordinateConstraint(bodyNumbers=[None, oBoom0List[0]], coordinates=[None, 0])
    mbs.CreateCoordinateConstraint(bodyNumbers=[None, oBoom0List[0]], coordinates=[None, 1])
    mbs.CreateCoordinateConstraint(bodyNumbers=[None, oBoom0List[0]], coordinates=[None, 2])
    mbs.CreateCoordinateConstraint(bodyNumbers=[None, oBoom0List[0]], coordinates=[None, 3])

#++++++++++++++++++++++++++++++++++++++++++++++++++
# add two sliding joints
offsetList = []
for i in range(nElements):
    offsetList += [abs((i-1)*elementLength)]
# fixed to end of boom1 and sliding on boom0
nGenericJSliding0 = mbs.AddNode(NodeGenericData(numberOfDataCoordinates=2,
                                                initialCoordinates=[0, 0]))
oJSliding0 = mbs.AddObject(ObjectJointSliding2D(markerNumbers=[mNBoom1E, mBBoom0List[1]], # we initialize the boom at second marker of midline
                                                slidingMarkerNumbers=mBBoom0List, # used to update sliding marker
                                                slidingMarkerOffsets=offsetList, # set offsets accordingly such that slider is at second marker
                                                nodeNumber=nGenericJSliding0,
                                                classicalFormulation=False))
# fixed to end of boom0 and sliding on boom1
nGenericJSliding1 = mbs.AddNode(NodeGenericData(numberOfDataCoordinates=2,
                                                initialCoordinates=[0, 0]))
oJSliding1 = mbs.AddObject(ObjectJointSliding2D(markerNumbers=[mNBoom0E, mBBoom1List[1]], # we initialize the midline at second marker of boom
                                                slidingMarkerNumbers=mBBoom1List, # used to update sliding marker
                                                slidingMarkerOffsets=offsetList, # set offsets accordingly such that slider is at second marker
                                                nodeNumber=nGenericJSliding1,
                                                classicalFormulation=False))

#++++++++++++++++++++++++++++++++++++++++++++++++++
# add load to the start of the boom1 and also a sensor measuring the displacement
lTip = mbs.AddLoad(LoadForceVector(markerNumber=mNBoom1S, loadVector=[0, loadTip, 0]))
sTip = mbs.AddSensor(SensorMarker(markerNumber=mNBoom1S, outputVariableType=exu.OutputVariableType.Position, writeToFile=False, storeInternal=True))

#++++++++++++++++++++++++++++++++++++++++++++++++++
# add coordinates constraint that extracts boom0 every transitionStayTime
oCCBoom0 = mbs.CreateCoordinateConstraint(bodyNumbers=[None, oBoom1List[0]],
                                          coordinates=[None, 0])
def PreStepUserFunction(mbs, t):
    if t > totalExperimentTime:
        x = xList[-1]
    else:
        iSegment = int(t/segmentTime)
        timeInSegment = t % segmentTime
        xStart = xList[iSegment]
        if xList[iSegment] == xList[-1]:
            xEnd = xList[-1]
        else:
            xEnd = xList[iSegment+1]
        x = xStart
        if timeInSegment > transitionStayTime:
            x = SmoothStep(timeInSegment-transitionStayTime, 0, transitionTime, xStart, xEnd)
    xOffset = x - elementLength # offset to reference configuration
    # print('Extraction length ' + str(xOffset) + 'm')
    mbs.SetObjectParameter(oCCBoom0, 'offset', xOffset)
    return True
mbs.SetPreStepUserFunction(PreStepUserFunction)

#%% do simulation (implicit dynamic simulation) (WYSWYS - What You See is What You Simulate)
mbs.Assemble()
h = 1e-3
tEnd = totalExperimentTime + 5

#++++++++++++++++++++++++++++++++++++++++++++++++++
# general visualization settings
SC.visualizationSettings.nodes.show = True
SC.visualizationSettings.markers.show = True
SC.visualizationSettings.markers.drawSimplified = True
SC.visualizationSettings.general.drawCoordinateSystem = True

SC.visualizationSettings.openGL.multiSampling = 4

#++++++++++++++++++++++++++++++++++++++++++++++++++
# simulation settings
simulationSettings = exu.SimulationSettings()
# simulationSettings.linearSolverType = exu.LinearSolverType.EigenSparse
simulationSettings.timeIntegration.numberOfSteps = int(tEnd/h)
simulationSettings.timeIntegration.endTime = tEnd
simulationSettings.solutionSettings.writeSolutionToFile = useSolutionViewer
simulationSettings.solutionSettings.solutionWritePeriod = 0.05
simulationSettings.timeIntegration.newton.useModifiedNewton = True
simulationSettings.displayComputationTime = True
# simulationSettings.solutionSettings.recordImagesInterval = h
simulationSettings.timeIntegration.verboseMode = 0

#++++++++++++++++++++++++++++++++++++++++++++++++++
if useRenderer:
    SC.renderer.Start()
    SC.renderer.DoIdleTasks()
mbs.SolveDynamic(simulationSettings=simulationSettings, showHints=True)
if useRenderer:
    SC.renderer.DoIdleTasks()
    SC.renderer.Stop() #safely close rendering window!

#++++++++++++++++++++++++++++++++++++++++++++++++++
if useSolutionViewer:
    mbs.SolutionViewer()
    
#%% print results
from plotter import setMatplotlibSettings
setMatplotlibSettings()
sData = mbs.GetSensorStoredData(sTip)
t = sData[:, 0]
xPos = sData[:, 1]
yPos = sData[:, 2]
import matplotlib.pyplot as plt
fig, ax1 = plt.subplots(1, 1, figsize=[8, 4])
ax1.plot(t, yPos, color='blue', label='y')
ax1.set_xlabel('Time in s')
ax1.set_ylabel('y displacement of tip (boom1) in m', color='blue')
ax1.tick_params(axis='y', colors='blue')
ax1.grid()
ax2 = ax1.twinx()
ax2.plot(t, xPos, color='red', label='x')
ax2.set_ylabel('x displacement of tip (boom1) in m', color='red')
ax2.tick_params(axis='y', colors='red')
# ax2.grid()
# lines1, labels1 = ax1.get_legend_handles_labels()
# lines2, labels2 = ax2.get_legend_handles_labels()
# ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
plt.tight_layout()
plt.show()
