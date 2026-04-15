#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#
# Details:  
# Authors:  Sebastian Weyrer
# Date:     2025-11-11
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import time
import exudyn as exu
from exudyn import graphics
from exudyn.utilities import *
from exudyn.FEM import *
import ngsolve as ngs
import netgen.occ as occ
from femDevelopment import MainSystemCreateFFRFReducedOrderObject
print('EXUDYN version:', exu.config.Version())

#%% define parameters and behavior of script
nModes = 6          # number of eigen modes in adition to static modes
useRenderer = True
useSolutionViewer = False

#%% set the parameters of the boom's geometry
height0 = 0.5           # [m] height of boom0
height1 = 0.4           # [m] height of boom1
length = 2              # [m] length of booms
meshOrder = 1
boom1Offset = 0.2
xList = [0.5, 0.75, 1, 1.25, 1.5]    # [m] extraction length of boom 1 where 0 is reference

#%% define the materials of the stick (here, we only use one material)
materials = {'rubber':{'youngsModulus':5e6, 'poissonsRatio':0.49, 'density':920}}

#%% make fem objects for boom0 with different BCs
modeBaseList = []
boom0FemObjectList = []
boundaryNodesListList = []
boundaryWeightsListList = []
for i, x in enumerate(xList):
    # make boom0 geometry and mesh
    boom0 = occ.Box((0, -height0/2, -height0/2), (length, height0/2, height0/2))
    boom0Subtract = occ.Box((boom1Offset, -height1/2, -height1/2), (length, height1/2, height1/2))
    boom0 -= boom0Subtract
    boom0.name = 'rubber' # assign material to geometries
    geo = occ.OCCGeometry(boom0)
    print('Start meshing geometry')
    start_time = time.time()
    mesh = ngs.Mesh(geo.GenerateMesh(maxh=0.1)) # note: per default tetrahedron elements are built
    print('Meshing needed %.3f seconds' % (time.time() - start_time))
    print('Number of elements:', mesh.ne)
    print('Used materials:', mesh.GetMaterials())
    if False: # this should only run one time so it is placed below 'if False'
        import netgen.gui
        ngs.Draw(mesh, clipping=True)
    # make EXUDYN's FEM object
    fem = FEMinterface()
    [bfM, bfK, fes] = fem.ImportMeshFromNGsolve(mesh, materials=materials,
                                                createBoundaryNodeSets=False,
                                                meshOrder=meshOrder)
    # make the boundary the 'old school' way as we need same boom discretization but different boundary nodes
    epsilon = (height0/2-height1/2) / 2
    boundaryNodes = fem.GetNodesInCube([boom1Offset+x, -height1/2-epsilon, -height1/2-epsilon], [length, height1/2+epsilon, height1/2+epsilon])
    boundaryWeights = fem.GetNodeWeightsFromSurfaceAreas(nodeList=boundaryNodes)
    boundaryNodesListList += [[boundaryNodes]]
    boundaryWeightsListList += [[boundaryWeights]]
    fem.ComputeHurtyCraigBamptonModes(boundaryNodesList=[boundaryNodes],
                                      nEigenModes=6,
                                      boundaryNodesWeights=[boundaryWeights],
                                      # computationMode=HCBstaticModeSelection.allBoundaryNodes,
                                      computationMode=HCBstaticModeSelection.RBE2,
                                      )
    boom0FemObjectList += [fem]
    modeBaseList += [fem.modeBasis['matrix']] # psi
    
#%% interpolate between two snapshots
from scipy.linalg import svd, pinv
modeBasis0 = modeBaseList[0] # reference basis
modeBasisN = modeBaseList[-1] # parameter point
# just use two snapshots and interpolate between them
snapshotList = [modeBasis0, modeBasisN]
# singular value decomposition, i.e. a == U @ S @ Vh
def GrassmannLog(Psi0, Psi):
    matrixToDecompose = (np.eye(Psi0.shape[0])-Psi0@Psi0.T) @ Psi @ pinv(Psi0.T@Psi)
    U, s, Vt = svd(matrixToDecompose, full_matrices=False)
    sigma = np.diag(s)
    gamma = U @ np.arctan(sigma) @ Vt
    return gamma
def GrassmannExp(Psi0, Gamma):
    U, s, Vt = svd(Gamma, full_matrices=False)
    sigma = np.diag(s)
    Psi = Psi0 @ Vt.T @ np.cos(sigma) + U @ np.sin(sigma)
    return Psi
# check the grassmann log and exp function
modeBasisNGrassmann = GrassmannExp(modeBasis0, GrassmannLog(modeBasis0, modeBasisN))
errorGrassmann = np.sum(np.abs(modeBasisNGrassmann - modeBasisN))
print('Grasssmann function has an error of', errorGrassmann)
# flatten snapshots with logarithmic map to gamma
gammaList = []
for snapshot in snapshotList:
    gammaList += [GrassmannLog(modeBasis0, snapshot)]
print(len(gammaList), 'flattened snapshots')
modeBasisIntList = []
xFractionList = []
for x in xList:
    xFractionList += [(x-xList[0]) / (xList[-1]-xList[0])]
    print(1-xFractionList[-1], 'weighted first snapshot')
    print(xFractionList[-1], 'weighted last snapshot')
    # now do interpolation for the xList just using the first and last snapshot
    gammaInt = (1-xFractionList[-1]) * gammaList[0] + xFractionList[-1] * gammaList[-1]
    modeBasisIntList += [GrassmannExp(modeBasis0, gammaInt)]
print('Evaluated relative x-positions:', xFractionList)

#%% now compare the mode bases
errorList = []
for i, xFraction in enumerate(xFractionList):
    modeBasisInt = modeBasisIntList[i]
    modeBasisI = modeBaseList[i]
    errorList += [np.linalg.norm(modeBasisI - modeBasisInt, 'fro')]
    print('Error for x fraction of', str(xFraction) + ':\t', errorList[-1])
    
#%% and now make FFRF objects for the booms
SC = exu.SystemContainer()
mbs = SC.AddSystem()
for i, boom0 in enumerate(boom0FemObjectList):
    cms = ObjectFFRFreducedOrderInterface(boom0)
    if i == 0 or i == (len(boom0FemObjectList) - 1):
        color = graphics.color.red
    else:
        color = graphics.color.green
    objFFRF = cms.AddObjectFFRFreducedOrder(mbs, positionRef=[0, -i, 0],
                                                 initialVelocity=[0, 0, 0], 
                                                 initialAngularVelocity=[0, 0, 0],
                                                 # stiffnessProportionalDamping=0.01,
                                                 rotationMatrixRef=RotationMatrixZ(0), # rotation w.r.t. reference position
                                                 color=color,
                                                 # gravity=[0, g, 0]
                                                 )
    for j, nodeSet in enumerate(boundaryNodesListList[i]):
        weights = boundaryWeightsListList[i][j] # these factors could be based on surface integrals of the constrained mesh faces
        marker = mbs.AddMarker(MarkerSuperElementRigid(bodyNumber=objFFRF['oFFRFreducedOrder'],
                                                       meshNodeNumbers=nodeSet,
                                                       weightingFactors=weights))

#%% do simulation (implicit dynamic simulation) (WYSWYS - What You See is What You Simulate)
mbs.Assemble()
h = 1e-3
tEnd = 1

#++++++++++++++++++++++++++++++++++++++++++++++++++
# general visualization settings
SC.visualizationSettings.nodes.show = False
SC.visualizationSettings.markers.show = True
SC.visualizationSettings.markers.defaultSize = 0.08 # to show what is happening
SC.visualizationSettings.markers.drawSimplified = False
SC.visualizationSettings.general.drawCoordinateSystem = False
SC.visualizationSettings.openGL.clippingPlaneNormal = [0, 0, 1] # add clipping plane

#++++++++++++++++++++++++++++++++++++++++++++++++++
# simulation settings
simulationSettings = exu.SimulationSettings()
simulationSettings.timeIntegration.numberOfSteps = int(tEnd/h)
simulationSettings.timeIntegration.endTime = tEnd
simulationSettings.solutionSettings.writeSolutionToFile = useSolutionViewer
simulationSettings.timeIntegration.newton.useModifiedNewton = True
simulationSettings.displayComputationTime = True
# simulationSettings.solutionSettings.recordImagesInterval = 0.05

#++++++++++++++++++++++++++++++++++++++++++++++++++
if useRenderer:
    SC.renderer.Start()
    SC.renderer.DoIdleTasks()
mbs.SolveDynamic(simulationSettings=simulationSettings)
if useRenderer:
    SC.renderer.DoIdleTasks()
    SC.renderer.Stop() #safely close rendering window!
    
#++++++++++++++++++++++++++++++++++++++++++++++++++
if useSolutionViewer:
    mbs.SolutionViewer()
