#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#
# Details:  Utilities for the 'BoomPMR'-Project.
# Author:   Sebastian Weyrer
# Date:     2026-04-22
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import pickle
import exudyn as exu
import exudyn.graphics as graphics
from exudyn.utilities import *
import matplotlib as mpl
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt

def GetStandardValues(nGlobalModes=None):
    standardColors = [graphics.color.blue, graphics.color.orange, [0.172549, 0.627451, 0.172549, 1.0], graphics.color.red] # colors of beams (also used for plots)
    if nGlobalModes is not None:
        labelList = ['ANCF', 'non-reduced (with AE)', 'globally reduced (r=' + str(nGlobalModes) + ') (with AE)', 'non-reduced (minimal coordiantes)']
    else:
        labelList = ['ANCF', 'non-reduced (with AE)', 'globally reduced (with AE)', 'non-reduced (minimal coordiantes)']
    rawDataLabelList = ['displ. x in m', 'displ. y in m', 'rot in rad']
    markerList = ['-o', '-*', '-p', '-+']
    timeStr = 'time in s'
    slidingStr = 'sliding phase'
    return standardColors, labelList, rawDataLabelList, markerList, timeStr, slidingStr

def SaveFig(figure, path, name, closeFig=False, saveFig=True):
    if saveFig:
        figure.savefig(path / f'{name}.pdf')
        figure.savefig(path / f'{name}.png', dpi=300)
    if closeFig:
        plt.close('all')

def SetMatplotlibSettings():
    fsize = 9
    tsize = 8
    tdir = 'out'
    major = 5.0
    minor = 3.0
    lwidth = 0.8
    lhandle = 2
    plt.style.use('default')
    plt.rcParams['font.family'] = 'serif'
    cmfont = font_manager.FontProperties(fname=mpl.get_data_path() + '/fonts/ttf/cmr10.ttf')
    plt.rcParams['font.serif'] = cmfont.get_name()
    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = fsize
    plt.rcParams['legend.fontsize'] = tsize
    plt.rcParams['xtick.direction'] = tdir
    plt.rcParams['ytick.direction'] = tdir
    plt.rcParams['xtick.major.size'] = major
    plt.rcParams['xtick.minor.size'] = minor
    plt.rcParams['ytick.major.size'] = major
    plt.rcParams['ytick.minor.size'] = minor
    plt.rcParams['xtick.major.pad'] = 4
    plt.rcParams['ytick.major.pad'] = 4
    plt.rcParams['axes.linewidth'] = lwidth
    plt.rcParams['axes.formatter.use_mathtext'] = True
    plt.rcParams['legend.handlelength'] = lhandle
    plt.rcParams['lines.linewidth'] = 1.2
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['figure.autolayout'] = True
    plt.rcParams['figure.dpi'] = 150

def PrintHeadlineToConsole(length=80, text=None):
        if not text:
            print('+' * length)
            return
        middle = f' {text} '
        if len(middle) >= length:
            print(middle)
            return
        remaining = length - len(middle)
        left = remaining // 2
        right = remaining - left
        print('+' * left + middle + '+' * right)
        
def GetGraphicsOfConstraint(position=[0, 0, 0]):
    constraintWidth = 0.3
    constraintHeight = 0.3
    sphereRadius = 0.07
    graphicsData = []
    graphicsData += [graphics.Quad(pList=[[-constraintWidth/2 + position[0], position[1]-constraintHeight, position[2]],
                                          [constraintWidth/2 + position[0], position[1]-constraintHeight, position[2]],
                                          [position[0], position[1], position[2]],
                                          [position[0], position[1], position[2]]], color=graphics.color.black)]
    graphicsData += [graphics.Sphere(point=[position[0], position[1], position[2]], radius=sphereRadius, color=graphics.color.black)]
    return graphicsData

def GetGraphicsOfConstraintsAndAddBasis(mbs, objectNumber, SlidingJointFunction,
                                        zOffsetForDrawing=0.01, yReferenceOffsetANCF=0, yReferenceOffsetBeamNR=0, yReferenceOffsetBeamPMR=0,
                                        xTextOffset=0, yTextOffset=0):
    graphicsData = []
    t = mbs.systemData.GetTime(exu.ConfigurationType.Visualization)
    # show constraints
    graphicsData += GetGraphicsOfConstraint(position=[0, yReferenceOffsetANCF, zOffsetForDrawing])
    graphicsData += GetGraphicsOfConstraint(position=[SlidingJointFunction(t), yReferenceOffsetANCF, zOffsetForDrawing])
    graphicsData += GetGraphicsOfConstraint(position=[0, yReferenceOffsetBeamNR, zOffsetForDrawing])
    graphicsData += GetGraphicsOfConstraint(position=[SlidingJointFunction(t), yReferenceOffsetBeamNR, zOffsetForDrawing])
    # if addSpecificallyReducedBeam:
    #     graphicsData += GetGraphicsOfConstraint(position=[0, yReferenceOffsetBeamSR, zOffsetForDrawing])
    #     graphicsData += GetGraphicsOfConstraint(position=[SlidingJointFunction(t), yReferenceOffsetBeamSR, zOffsetForDrawing])
    graphicsData += GetGraphicsOfConstraint(position=[0, yReferenceOffsetBeamPMR, zOffsetForDrawing])
    graphicsData += GetGraphicsOfConstraint(position=[SlidingJointFunction(t), yReferenceOffsetBeamPMR, zOffsetForDrawing])
    # add bases
    graphicsData += [graphics.Basis(origin=[0, yReferenceOffsetANCF, 0], length=0.5)]
    graphicsData += [graphics.Basis(origin=[0, yReferenceOffsetBeamNR, 0], length=0.5)]
    # if addSpecificallyReducedBeam:
    #     graphicsData += [graphics.Basis(origin=[0, yReferenceOffsetBeamSR, 0], length=0.5)]
    graphicsData += [graphics.Basis(origin=[0, yReferenceOffsetBeamPMR, 0], length=0.5)]
    graphicsData += [graphics.Text(point=[xTextOffset, yReferenceOffsetANCF+yTextOffset, 0], text='ANCF')]
    graphicsData += [graphics.Text(point=[xTextOffset, yReferenceOffsetBeamNR+yTextOffset, 0], text='non-reduced')]
    # if addSpecificallyReducedBeam:
    #     graphicsData += [graphics.Text(point=[xTextOffset, yReferenceOffsetBeamSR+yTextOffset, 0], text='spec. reduced')]
    graphicsData += [graphics.Text(point=[xTextOffset, yReferenceOffsetBeamPMR+yTextOffset, 0], text='globally reduced (ours)')]
    return graphicsData

# helper function that adds ANCF 2D Beam with sliding joint and requested sensors to a mbs
# this function is created to avoid code duplication, as different ANCF beams are added for different tests
def AddANCFBeamWithSlidingJointToMBSAndAddSensors(mbs, SlidingJointFunction, LoadTipFunction, length=1, rhoA=50, EI=1e6, EA=1e6,
                                                  nElementsANCF=5, yReferenceOffsetANCF=0, width=0.5, show=False,
                                                  mCoordGround=None, sensorOutputVariableTypeANCFList=[], beamNodeReferencePositionList=None):
    standardColors, _, _, _, _, _ = GetStandardValues()
    
    #++++++++++++++++++++++++++++++++++++++++++++++++++
    # setup of ANCF beam (S...Start of element, E...End of element)
    elementLengthANCF = length/nElementsANCF
    nSANCF = mbs.AddNode(NodePoint2DSlope1(referenceCoordinates=[0, yReferenceOffsetANCF, 1, 0],
                                           initialCoordinates=[0]*4, initialVelocities=[0]*4))
    # add two markers here to (later) constrain the position of the beginning of the ANCF beam
    mSANCFX = mbs.AddMarker(MarkerNodeCoordinate(nodeNumber=nSANCF, coordinate=0))
    mSANCFY = mbs.AddMarker(MarkerNodeCoordinate(nodeNumber=nSANCF, coordinate=1))
    mANCFList = [] # list of body markers to ANCF beam for sliding joint
    xPositionsOfSensors = beamNodeReferencePositionList # the sensors of the ANCF beam should be at the nodes of the other beam
    sDictANCF = {} # special dict to add sensors
    for i in range(nElementsANCF):
        nEANCF = mbs.AddNode(NodePoint2DSlope1(referenceCoordinates=[(elementLengthANCF*(i+1)), yReferenceOffsetANCF, 1, 0],
                                               initialCoordinates=[0]*4, initialVelocities=[0]*4))
        oANCF = mbs.AddObject(ObjectANCFCable2D(nodeNumbers=[nSANCF, nEANCF],
                                                physicsLength=elementLengthANCF,
                                                physicsMassPerLength=rhoA,
                                                physicsBendingStiffness=EI,
                                                physicsAxialStiffness=EA,
                                                visualization=VObjectALEANCFCable2D(drawHeight=width, color=standardColors[0], show=show)))
        mANCFList += [mbs.AddMarker(MarkerBodyCable2DCoordinates(bodyNumber=oANCF))]
        nSANCF = nEANCF # start is end of last element
        
        #++++++++++++++++++++++++++++++++++++++++++++++++++
        # special logic to add required sensors acc. to sensorOutputVariableTypeANCFList
        # compute the local position of the sensors and add them
        if i == nElementsANCF - 1:
            globalXPositionOfSensorList = xPositionsOfSensors[(xPositionsOfSensors >= elementLengthANCF*i) & (xPositionsOfSensors <= elementLengthANCF*(i+1))]
        else:
            globalXPositionOfSensorList = xPositionsOfSensors[(xPositionsOfSensors >= elementLengthANCF*i) & (xPositionsOfSensors < elementLengthANCF*(i+1))]
        sDictElementANCF = {}
        sDictANCF['element' + str(i)] = sDictElementANCF
        for outputVariableType in sensorOutputVariableTypeANCFList:
            sDictLocalPosition = {}
            sDictElementANCF[str(outputVariableType)] = sDictLocalPosition
            for globalXPosition in globalXPositionOfSensorList:
                localXPosition = globalXPosition - (elementLengthANCF*i)
                sDictLocalPosition[str(localXPosition)] = mbs.AddSensor(SensorBody(bodyNumber=oANCF, localPosition=[localXPosition, 0, 0],
                                                                                   outputVariableType=outputVariableType,
                                                                                   storeInternal=True))
    #++++++++++++++++++++++++++++++++++++++++++++++++++
    # add marker to apply load (only in y-direction), apply load
    mTipANCF = mbs.AddMarker(MarkerNodeCoordinate(nodeNumber=nEANCF, coordinate=1))
    mbs.AddLoad(LoadCoordinate(markerNumber=mTipANCF, loadUserFunction=LoadTipFunction))
    
    #++++++++++++++++++++++++++++++++++++++++++++++++++
    # now constrain the beginning of ANCF beam (we can not use create function since we do not constrain object but specific node)
    mbs.AddObject(CoordinateConstraint(markerNumbers=[mCoordGround, mSANCFX], visualization=VObjectConnectorCoordinate(show=False)))
    mbs.AddObject(CoordinateConstraint(markerNumbers=[mCoordGround, mSANCFY], visualization=VObjectConnectorCoordinate(show=False)))
    
    #++++++++++++++++++++++++++++++++++++++++++++++++++
    # add mass point that is moved in order to make sliding joint (initialize acc. to sliding joint function)
    # acc. to https://exudyn.readthedocs.io/en/latest/docs/RST/items/ObjectJointSliding2D.html#objectjointsliding2d we need a point mass / rigid body marker here
    # the mass does not influence the solution: coordinates constraint (algebraically) moves it further ...
    # also add coordinates constraint
    # TODO: Ask Johannes: Maybe in this case there is a more convenient way to get the moving marker without adding a mass point 
    oMP = mbs.CreateMassPoint(physicsMass=1, referencePosition=[0, yReferenceOffsetANCF, 0],
                              initialDisplacement=[SlidingJointFunction(0), 0, 0], color=graphics.color.red)
    oCCMP = mbs.CreateCoordinateConstraint(bodyNumbers=[None, oMP], coordinates=[None, 0], show=False)
    mbs.CreateCoordinateConstraint(bodyNumbers=[None, oMP], coordinates=[None, 1], show=False)
    # here, we just use the sliding joint function in the PreSetpUserFunction
    def PreStepUserFunction(mbs, t):
        mbs.SetObjectParameter(oCCMP, 'offset', SlidingJointFunction(t))
        return True
    mbs.SetPreStepUserFunction(PreStepUserFunction)
    
    #++++++++++++++++++++++++++++++++++++++++++++++++++
    # add marker to mass point and add sliding joint
    mMP = mbs.AddMarker(MarkerBodyPosition(bodyNumber=oMP))
    # add sliding joint to beam
    offsetList = np.arange(nElementsANCF) * (length/nElementsANCF)
    nGenericJSliding = mbs.AddNode(NodeGenericData(numberOfDataCoordinates=2, initialCoordinates=[0, 0]))
    oJSliding = mbs.AddObject(ObjectJointSliding2D(markerNumbers=[mMP, mANCFList[0]],
                                                   slidingMarkerNumbers=mANCFList,
                                                   slidingMarkerOffsets=offsetList,
                                                   nodeNumber=nGenericJSliding,
                                                   classicalFormulation=False))
    return sDictANCF

def GetOutputDictANCFFromSDictANCF(mbs, sDictANCF={}, sensorOutputVariableTypeANCFList=[], verbose=False):
    sOrderedDictANCF = {}
    for ele in sDictANCF:
        for outputVariable in sDictANCF[ele]:
            for localPosition in sDictANCF[ele][outputVariable]:
                sOrderedDictANCF.setdefault(outputVariable, []).append(sDictANCF[ele][outputVariable][localPosition])
    outputDictANCF = {}
    nSensorsANCF = 0
    for outputVariable in sensorOutputVariableTypeANCFList:
        output = np.array([])
        for sensor in sOrderedDictANCF[str(outputVariable)]:
            nSensorsANCF += 1
            if np.size(output) == 0:
                output = mbs.GetSensorStoredData(sensorNumber=sensor)[:, 1:]
            else:
                output = np.hstack((output, mbs.GetSensorStoredData(sensorNumber=sensor)[:,1:]))
        outputDictANCF[str(outputVariable)] = output
    if verbose:
        print('For the ANCF element, a total of', nSensorsANCF, 'sensors are added.')
    return outputDictANCF

def GetKineticAndStrainAndPotentialANCFEnergyFromOutputDictANCF(outputDictANCF, nTimeSteps=100, length=1, rhoA=40,
                                                                LoadTipFunction=None, nBeamElements=40, sensorsWritePeriod=1e-2):
    EANCF = np.zeros(nTimeSteps)
    deltaX = length/nBeamElements # used for trapezoidal rule to integrate over beam length
    for i in range(nTimeSteps):
        # kinetic energy
        velocityTranslationalSum = 0
        for j in range(nBeamElements): # which gives us the number of sensors -1 per oputputVariable
            velXLocal = outputDictANCF['OutputVariableType.Velocity'][i, 3*j] # x
            velXNext = outputDictANCF['OutputVariableType.Velocity'][i, 3*(j+1)] # x
            velYLocal = outputDictANCF['OutputVariableType.Velocity'][i, 3*j+1] # y
            velYNext = outputDictANCF['OutputVariableType.Velocity'][i, 3*(j+1)+1] # y
            velocityTranslationalSum += (((velXLocal**2+velYLocal**2) + (velXNext**2+velYNext**2))/2) # translational
        velocityTranslationalSum *= deltaX # put that out of the sum
        EANCF[i] = (1/2) * rhoA * velocityTranslationalSum
        # unity check: (kg/m)*(m^2/s^2)*m = (kg*m)/s^2 * m = Nm
        # strain energy
        strainSum = 0
        for j in range(nBeamElements): # which gives us the number of sensors -1 per oputputVariable
            curvatureLocal = outputDictANCF['OutputVariableType.CurvatureLocal'][i, j]
            curvatureLocalNext = outputDictANCF['OutputVariableType.CurvatureLocal'][i, j+1]
            torqueLocal = outputDictANCF['OutputVariableType.TorqueLocal'][i, j]
            torqueLocalNext = outputDictANCF['OutputVariableType.TorqueLocal'][i, j+1]
            strainSum += (((curvatureLocal*torqueLocal) + (curvatureLocalNext*torqueLocalNext))/2)
        strainSum *= deltaX # put that out of the sum
        # unity check: (1/m) * (Nm) * m = Nm
        EANCF[i] += (1/2) * strainSum
        # load potential
        EANCF[i] -= outputDictANCF['OutputVariableType.Displacement'][i, -2] * LoadTipFunction(None, i*sensorsWritePeriod, None)
    return EANCF

# helping fuction to compute Simple Moving Average (SMA)
def SMA(time, data, window, SMAMode):
    SMA = np.convolve(data, np.ones(window) / window, mode=SMAMode)
    if SMAMode == 'valid':
        timeSMA = time[window//2 : window//2 + len(SMA)] # trim the time for 'valid' mode
    else:
        timeSMA = time # do not trim
    return timeSMA, SMA

# helping function for the Normalized Root-Mean-Square Error (NRMSE) of a scalar signal
def NRMSE(signal, reference):
    rmse = np.sqrt(np.mean((signal - reference)**2))
    norm = np.max(reference) - np.min(reference)
    return rmse / norm

def RMSE(signal, reference):
    return np.sqrt(np.mean((signal - reference)**2))

def AbsError(signal, reference):
    return np.linalg.norm(signal-reference)

# helping function for the Normalized Root-Mean-Square Error (NRMSE) of a vector signal
def NRMSEVector(signal, reference):
    # signal, reference: [nTimeSteps, nNodes}
    norm = np.max(reference, axis=0) - np.min(reference, axis=0) # range over time, per node
    rmse = np.sqrt(np.mean((signal - reference)**2, axis=0)) # rmse over time, per node
    mask = norm > 1e-10
    return np.mean(rmse[mask] / norm[mask])

def GetStandardSimulationAndSetVisualizationSettings(tEnd, h, exportFrames=False):
    simulationSettings = exu.SimulationSettings()
    visualizationSettings = exu.VisualizationSettings()
    simulationSettings.timeIntegration.numberOfSteps = int(tEnd/h)
    simulationSettings.timeIntegration.endTime = tEnd
    simulationSettings.solutionSettings.writeSolutionToFile = False
    simulationSettings.timeIntegration.newton.useModifiedNewton = True
    simulationSettings.displayComputationTime = True
    simulationSettings.timeIntegration.verboseMode = 0
    visualizationSettings.general.autoFitScene = True
    if exportFrames:
        simulationSettings.solutionSettings.recordImagesInterval = h*20
    return simulationSettings

def AppendResultsToPKLFile(fileName, key, newResultsDict):
    with open(fileName, 'rb') as f:
        data = pickle.load(f)
    if key in data['results']:
        data['results'][key].update(newResultsDict)
    else:
        data['results'][key] = newResultsDict
    with open(fileName, 'wb') as f:
        pickle.dump(data, f)

def GetResultsDictFromPKLFile(fileName, key):
    with open(fileName, 'rb') as f:
        data = pickle.load(f)
    return data['results'][key]
    