#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#
# Details:  Script that computes the eigenmodes of a boom with different
#           Boundary Conditions (BCs) and collects snapshots for global and
#           local Parameterized Model Order Reduction (PMOR). The displacement
#           of the boom with BC configurations is computed for the (1) non
#           reduced boom, (2) the spec. reduced boom, i.e. a reduction basis is
#           built for each BC configuration, (3) reduction basis is
#           interpolated between the snapshots, i.e. local PMOR, (4)
#           globally reduced boom, i.e. global PMOR is used such that one
#           reduction basis is used for all BC configurations, and (5) a random
#           reduction basis as a sanity check of the code. Note the
#           difference between
#           - psiBC/KBC/MBC ... ; eigenmodes ... for the system with BC
#                                 enforced, i.e. cancled out rows/columns
#           - psiLifted ... ; eigenmodes ... for the system with BC enforced
#                             and constrained DOFs zeroed
#           - q; displacement in reduced (modal) coordinates
#           - u; displacement in non modal coordinates (BC or lifted)
#           We name matrices, vectors, etc. with 'configuration' if they
#           correspond to a specific configuration and remove this term when
#           speaking of quantities within the (stat./dyn.) equations. This
#           script also includes first dynamic tests.
# Author:   Sebastian Weyrer
# Date:     2026-01-29
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import os
import time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import eigh, solve, logm
import matplotlib.pyplot as plt
from plotter import SetMatplotlibSettings, getStandardColorTemplate
plt.close('all')

#%% define bahvior of script and the boom
# parameters of the boom
l = 1                                   # [m] beam length
E = 5e7#210e9                               # [N/m^2] E modulus
I = 1e-6                                # [m^4] second moment of area
rho = 7800/10                              # [kg/m^3] density
A = 1e-4                                # [m^2] cross-sectional area
fList = [0, -10, -20, -30]              # [N] list of different loads with which the tip of the boom is loaded
nElements = int(50)                     # number of elements the boom is made out of
# parameters for the PMOR
nSnapshots = 4                          # number of snapshots
configurationsMode = 2                  # mode how to sample configurations (0, 1, 2, 3, 4, 5)
nEigenmodesInSnapshots = 6              # number of eigenmodes computed per snapshot
nEigenmodesForSpecificReduction = 6     # number of eigenmodes used for specific reduction
energyTol = None                        # energy tolerance for truncation of global basis (either None or value); nGlobalModes must be None if set; standard: 0.999
nGlobalModes = 24                       # number of modes used in global basis (either None or value); energyTol must be None if set; highest: nSnapshots * nEigenmodesInSnapshots
saveFigures = False
playVideos = False
markevery = 5

#++++++++++++++++++++++++++++++++++++++++++++++++++
# automatically computed parameters
print('Parametric Model Order Reduction (PMOR) test script for a boom.')
if nGlobalModes is not None and nGlobalModes > (nSnapshots * nEigenmodesInSnapshots):
    print('The specified "nGlobalModes" is invalid, setting to', nSnapshots * nEigenmodesInSnapshots)
nForces = len(fList)
nNodes = nElements + 1
nodeNumbersList = np.arange(0, nNodes, 1)
nDOF = 2 * nNodes # displacement and rotation per node
lElement = l / nElements
nodePartsList = np.array_split(nodeNumbersList, nSnapshots) # split the beam into parts
print('++++++++++++++++++++++++++++++++++++++++++++++++++')
print('Number of nodes:\t\t\t', nNodes)
print('Number of elements:\t\t\t', nElements)
print('Degrees of Freedom (DOF):\t', nDOF)
# where should the snapshots points be placed and how many should be made?
constrainedNodeNumbersSnapshotsList = [] # node numbers constrained to get snapshots
for part in nodePartsList:
    constrainedNodeNumbersSnapshotsList += [part[-1]]
# if only one node is constrained, make a list for it
if type(constrainedNodeNumbersSnapshotsList) == np.int64:
    constrainedNodeNumbersSnapshotsList = np.array([constrainedNodeNumbersSnapshotsList])
print('++++++++++++++++++++++++++++++++++++++++++++++++++')
print('For', nSnapshots, 'snapshot(s) the displacement of the node(s)')
for constrainedNodeNumber in constrainedNodeNumbersSnapshotsList:
    print('->', constrainedNodeNumber)
print('will be constrained additionally.')
nSnapshots = len(constrainedNodeNumbersSnapshotsList)
print(nSnapshots, 'snapshot(s) are/is made.')
# where should the configuration points be placed and how many should be made?
nodeNumbersSnapshotDistance = int((constrainedNodeNumbersSnapshotsList[-1] - constrainedNodeNumbersSnapshotsList[-2])/2)
if configurationsMode == 0:
    constrainedNodeNumbersConfigurationsList = [constrainedNodeNumbersSnapshotsList[0] + nodeNumbersSnapshotDistance]
    constrainedNodeNumbersConfigurationsList += [constrainedNodeNumbersSnapshotsList[-2] + nodeNumbersSnapshotDistance]
elif configurationsMode == 1:
    constrainedNodeNumbersConfigurationsList = nodeNumbersList[1::2]
elif configurationsMode == 2:
    constrainedNodeNumbersConfigurationsList = nodeNumbersList[1:-1:1]
elif configurationsMode == 3:
    constrainedNodeNumbersConfigurationsList = nodeNumbersList[constrainedNodeNumbersSnapshotsList[0]::2]
elif configurationsMode == 4:
    constrainedNodeNumbersConfigurationsList = nodeNumbersList[constrainedNodeNumbersSnapshotsList[0]::1]
elif configurationsMode == 5:
    constrainedNodeNumbersConfigurationsList = nodeNumbersList[constrainedNodeNumbersSnapshotsList[0]]
# if only one node is constrained, make a list for it
if type(constrainedNodeNumbersConfigurationsList) == np.int64:
    constrainedNodeNumbersConfigurationsList = np.array([constrainedNodeNumbersConfigurationsList])
nConfigurations = len(constrainedNodeNumbersConfigurationsList)
print(nConfigurations, 'configuration(s) are/is investigated.')

#++++++++++++++++++++++++++++++++++++++++++++++++++
# initialize time counters
timeNonReducedList = [0, 0, 0, 0] # first entry is time to get reduction bases, second entry is solving linear equations, third entry is assembling global basis, fourth entry is interpolation
timeSpecReducedList = [0, 0, 0, 0] # first entry is time to get reduction bases, second entry is solving linear equations, third entry is assembling global basis, fourth entry is interpolation
timeIntReducedList = [0, 0, 0, 0] # first entry is time to get reduction bases, second entry is solving linear equations, third entry is assembling global basis, fourth entry is interpolation
timeGlobalReducedList = [0, 0, 0, 0] # first entry is time to get reduction bases, second entry is solving linear equations, third entry is assembling global basi, fourth entry is interpolations
timeRandomReducedList = [0, 0, 0, 0] # first entry is time to get reduction bases, second entry is solving linear equations, third entry is assembling global basis, fourth entry is interpolation

#%% define element matrices and assemble global matrices
def BeamElementMatrices(E, I, rho, A, l):
    k = (E*I/l**3) * np.array([[12,   6*l,  -12,   6*l],
                               [6*l, 4*l**2, -6*l, 2*l**2],
                               [-12, -6*l,   12,  -6*l],
                               [6*l, 2*l**2, -6*l, 4*l**2]])
    m = (rho*A*l/420) * np.array([[156,  22*l,   54,  -13*l],
                                  [22*l, 4*l**2, 13*l, -3*l**2],
                                  [54,   13*l,  156,  -22*l],
                                  [-13*l,-3*l**2,-22*l, 4*l**2]])
    return k, m

#++++++++++++++++++++++++++++++++++++++++++++++++++
# assemble global Finite Element (FE) stiffness and mass matrices such that M @ x_tt + K @ x = F
K = np.zeros([nDOF, nDOF])
M = np.zeros([nDOF, nDOF])
for e in range(nElements):
    ke, me = BeamElementMatrices(E, I, rho, A, lElement)
    indexList = [2*e, 2*e+1, 2*e+2, 2*e+3] # always 2 DOFs are overlapping
    for i in range(4):
        for j in range(4):
            K[indexList[i], indexList[j]] += ke[i, j]
            M[indexList[i], indexList[j]] += me[i, j]
            
#%% helping functions to compute eigenmodes, postprocess, and plot them
# function to compute eigenmodes for given constrainedNodesList
# the deflection of the first node is always automatically constrained
# K, M are the global stiffness- and mass matrices
# constrainedNodeNumbersList is a list for the (displacement) constrained nodes
# getBCMatrices is a flag; if True, the function resturns the FE matrices respecting the BCs
def GetUnconstrainedEigenmodesForConstrainedNodeNumbersList(K=np.array([]), M=np.array([]),
                                                            constrainedNodeNumbersList=[],
                                                            getBCMatrices=False, nEigenmodes=1):
    unconstrainedEigenmodesList = []
    KBCList = []
    MBCList = []
    freeDOFSList = []
    constrainedDOFSList = []
    computationTime = 0
    for constrainedNodeNumber in constrainedNodeNumbersList:
        constrainedDOFS = [0] # deflection of first node is constrained
        constrainedDOFS += [constrainedNodeNumber * 2] # deflection of node is constraint
        freeDOFS = np.setdiff1d(np.arange(nDOF), constrainedDOFS) # get list of free DOFs
        KBC = K[np.ix_(freeDOFS, freeDOFS)]
        MBC = M[np.ix_(freeDOFS, freeDOFS)]
        freeDOFSList += [freeDOFS]
        constrainedDOFSList += [constrainedDOFS]
        KBCList += [KBC]
        MBCList += [MBC]
        if not getBCMatrices:
            # solve the eigenvalues problem
            startTime = time.time()
            eigvals, eigvecs = eigh(KBC, MBC, subset_by_index=[0, nEigenmodes-1])
            endTime = time.time()
            computationTime += endTime - startTime
            unconstrainedEigenmodesList += [eigvecs]
    if getBCMatrices:
        return dict(KBCList=KBCList, MBCList=MBCList,
                    freeDOFSList=freeDOFSList)
    else:
        return dict(KBCList=KBCList, MBCList=MBCList,
                    freeDOFSList=freeDOFSList, constrainedDOFSList=constrainedDOFSList,
                    unconstrainedEigenmodesList=unconstrainedEigenmodesList, time=computationTime)

# function to fill constrained lines in a matrix with 0
# unconstrainedEigenmodes is the matrix where the constrained DOFs are filled with 0
# nDOF is the number of Degrees of Freedom of the system
# nEigenmodes is the number of eigenmodes
def GetConstrainedEigenmodesOutOfUnconstrainedEigenmodes(unconstrainedEigenmodes=np.array([]),
                                                         freeDOFS=[], nDOF=1, nEigenmodes=1):
    psi = np.zeros([nDOF, nEigenmodes])
    constrainedDOFSList = np.setdiff1d(np.arange(nDOF), freeDOFS)
    nSkipedDOFs = 0
    for DOF in range(nDOF):
        if DOF in freeDOFS:
            if unconstrainedEigenmodes.ndim == 1:
                psi[DOF, :] = unconstrainedEigenmodes[DOF-nSkipedDOFs]
            elif unconstrainedEigenmodes.ndim == 2:
                psi[DOF, :] = unconstrainedEigenmodes[DOF-nSkipedDOFs, :]
        if DOF in constrainedDOFSList:
            nSkipedDOFs += 1
    return psi

# function for post processing (sign-correction and global mass normalization) of a list of eigenmodes
# psiList is a list of the snapshots that should be equalized
def PostprocessEigenmodesList(M=M, psiList=[], nEigenmodes=3, verbose=True):
    nSnapshots = len(psiList)
    # do sign correction
    for i in range(1, nSnapshots):
        psiPrev = psiList[i-1]
        psiCurr = psiList[i]
        s = np.trace(psiPrev.T @ M @ psiCurr)
        if s < 0:
            psiCurr *= -1
        print('Changing sign of snaposhot', str(i+1) + '.')
        psiList[i] = psiCurr
    # do GLOBAL mass normalization
    for i, psi in enumerate(psiList):
        psiMassNormalized = np.zeros_like(psi)
        for j in range(nEigenmodes):
            normFactor = np.sqrt(psi[:, j].T @ M @ psi[:, j])
            psiMassNormalized[:, j] = psi[:, j] / normFactor
        # if verbose:
        #     for j in range(nEigenmodes):
        #         print('Mass norm. check:', round(psiMassNormalized[:, j].T @ M @ psiMassNormalized[:, j], 2))
        psiList[i] = psiMassNormalized
    return psiList

# function for plotting the eigenmodes and constrained nodes
# psiList is a list of the snapshots that should be plotted
# constrainedNodeNumbersList is a list for the (displacement) constrained nodes
# nodeNumbersList is a list of the node numbers of the beam
# nEigenmodesToPlot is the number of eigenmodes that should be plotted
# figsize is the size of the figure
def PlotEigenmodes(psiList=[], constrainedNodeNumbersList=[], nodeNumbersList=[],
                   nEigenmodesToPlot=3, figsize=[8, 8]):
    SetMatplotlibSettings()
    nSnapshots = len(psiList)
    fig, axs = plt.subplots(nSnapshots, 2, sharex=True, figsize=figsize)
    if nSnapshots == 1:
        axs = np.array([axs])
    linesForLegend = []
    labelsForLegend = []
    for i in range(nSnapshots):
        for j in range(nEigenmodesToPlot):
            line, = axs[i, 0].plot(nodeNumbersList, psiList[i][::2, j])
            line, = axs[i, 1].plot(nodeNumbersList, psiList[i][1::2, j])
            if i == 0:
                linesForLegend.append(line)
                labelsForLegend.append('mode' + str(j+1))
        axs[i, 0].plot(0, 0, 'o', color='k')
        axs[i, 0].plot(constrainedNodeNumbersList[i], 0, 'o', color='k')
        axs[i, 1].plot(0, 0, 'o', color='k')
        axs[i, 1].plot(constrainedNodeNumbersList[i], 0, 'o', color='k')
        axs[i, 0].grid()
        axs[i, 1].grid()
        axs[i, 0].set_ylabel('def. component')
        axs[i, 1].set_ylabel('rot. component')
    fig.legend(linesForLegend, labelsForLegend, loc='upper center',
               ncol=nEigenmodesToPlot, bbox_to_anchor=(0.5, 0.97))
    axs[-1, 0].set_xlabel('node number')
    axs[-1, 1].set_xlabel('node number')
    plt.subplots_adjust(right=0.8, left=0.2, top=0.9, wspace=0.6)

# function to plot three displacements and animate additionally
# fList is the list of forces that is applied
# constrainedNodeNumbersList is a list to draw the BCs accordingly
# nodeNumbersList is the x-coordinate
# figsize is the size of the figure
# showLegend determines whether legend is shown
# animate determines whether an animation is shown
# pauseTime is the time in second between two frames
# saveFigures determines whether .png files are written during animation (to make video)
def PlotDisplacement(displacementNonReducedList=[],
                     displacementSpecReducedList=[],
                     displacementGlobalReducedList=[],
                     fList=[], constrainedNodeNumbersList=[],
                     nodeNumbersList=[], figsize=[8, 8],
                     showLegend=True, animate=False, pauseTime=0.1, saveFigures=False):
    SetMatplotlibSettings()
    colors = getStandardColorTemplate()
    nSnapshots = len(displacementNonReducedList)
    nForces = len(displacementNonReducedList[0])
    if saveFigures:
        os.makedirs("images", exist_ok=True)
    if animate:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        for i in range(nSnapshots):
            ax.clear()
            for j in range(nForces):
                ax.plot(nodeNumbersList,
                        displacementNonReducedList[i][j][::2],
                        color=colors[j],
                        label='f = ' + str(fList[j]*(-1)) + 'N; non reduced')
                if displacementSpecReducedList != []:
                    ax.plot(nodeNumbersList,
                            displacementSpecReducedList[i][j][::2],
                            marker='*',
                            markevery=markevery,
                            color=colors[j],
                            label='f = ' + str(fList[j]*(-1)) + 'N; spec. reduced')
                if displacementGlobalReducedList != []:
                    ax.plot(nodeNumbersList,
                            displacementGlobalReducedList[i][j][::2],
                            marker='p',
                            markevery=markevery,
                            color=colors[j],
                            label='f = ' + str(fList[j]*(-1)) + 'N; global reduced')
            ax.plot(0, 0, 'o', color='k')
            ax.plot(constrainedNodeNumbersList[i], 0, 'o', color='k')
            ax.grid()
            if showLegend:
                ax.legend()
            ax.set_xlabel('node number')
            ax.set_ylabel('displacement in m')
            ax.set_ylim([-2e-5, 1e-5])
            plt.tight_layout()
            if saveFigures:
                plt.savefig('images/frame'+str(i)+'.png')
            plt.pause(pauseTime)
        plt.show()
        return
    fig, axs = plt.subplots(nSnapshots, 1, sharex=True, figsize=figsize)
    if nSnapshots == 1:
        axs = np.array([axs])
    for i in range(nSnapshots):
        for j in range(nForces):
            axs[i].plot(nodeNumbersList,
                        displacementNonReducedList[i][j][::2],
                        color=colors[j],
                        label='f = ' + str(fList[j]*(-1)) + 'N; non reduced')
            if displacementSpecReducedList != []:
                axs[i].plot(nodeNumbersList,
                            displacementSpecReducedList[i][j][::2],
                            marker='*',
                            markevery=markevery,
                            color=colors[j],
                            label='f = ' + str(fList[j]*(-1)) + 'N; spec. reduced')
            if displacementGlobalReducedList != []:
                axs[i].plot(nodeNumbersList,
                            displacementGlobalReducedList[i][j][::2],
                            marker='p',
                            markevery=markevery,
                            color=colors[j],
                            label='f = ' + str(fList[j]*(-1)) + 'N; global reduced')
        axs[i].plot(0, 0, 'o', color='k')
        axs[i].plot(constrainedNodeNumbersList[i], 0, 'o', color='k')
        axs[i].grid()
        axs[i].set_ylabel('displacement in m')
        if showLegend:
            axs[0].legend()
    axs[-1].set_xlabel('node number')
    plt.tight_layout()

# standard linear interpolation between two matrices gammaPrevious and gammaFollowing
# parameterRequested is the parameter at whcih the interpolated matrix is
def StandardInterpolationInTangentSpace(gammaPrevious, gammaFollowing,
                                        parameterPrevious, parameterFollowing, parameterRequested):
    alpha = (parameterRequested - parameterPrevious) / (parameterFollowing - parameterPrevious)
    # if i.e. parameterRequested = parameterPrevious, alpha must be 0 to select just the previous one
    # if the requested parameter is at the following, alpha is 1
    return (1 - alpha) * gammaPrevious + alpha * gammaFollowing

#%% compute eigenvalues and eigenvectors for snapshots
snapshotsDict = GetUnconstrainedEigenmodesForConstrainedNodeNumbersList(K=K, M=M, constrainedNodeNumbersList=constrainedNodeNumbersSnapshotsList,
                                                                        nEigenmodes=nEigenmodesInSnapshots)    
timeGlobalReducedList[0] += snapshotsDict['time']
timeIntReducedList[0] += snapshotsDict['time']
psiBCSnapshotsList = snapshotsDict['unconstrainedEigenmodesList']
freeDOFSSnapshotsList = snapshotsDict['freeDOFSList']
# now also get the constrained entries in the eigenvectors to get psi matrix
psiLiftedSnapshotsList = []
for i, psiBCSnapshot in enumerate(psiBCSnapshotsList):
    freeDOFS = freeDOFSSnapshotsList[i]
    psiLiftedSnapshotsList += [GetConstrainedEigenmodesOutOfUnconstrainedEigenmodes(unconstrainedEigenmodes=psiBCSnapshot,
                                                                              nDOF=nDOF, nEigenmodes=nEigenmodesInSnapshots,
                                                                              freeDOFS=freeDOFS)]
# ... and postprocess this matrices
psiLiftedSnapshotsList = PostprocessEigenmodesList(psiList=psiLiftedSnapshotsList, M=M, nEigenmodes=nEigenmodesInSnapshots)
if nSnapshots < 6:
    if nEigenmodesInSnapshots >= 3:
        nEigenmodesToPlot = 3
    else:
        nEigenmodesToPlot = nEigenmodesInSnapshots
    PlotEigenmodes(psiList=psiLiftedSnapshotsList,
                   constrainedNodeNumbersList=constrainedNodeNumbersSnapshotsList,
                   nodeNumbersList=nodeNumbersList, nEigenmodesToPlot=nEigenmodesToPlot)

#%% assemble one global reduction basis (psiLiftedGlobal) using the lifted and postprocessed eigenmodes of the snapshots
for i, psiLifted in enumerate(psiLiftedSnapshotsList):
    if i == 0:
        psiLiftedConc = psiLifted
    else:
        psiLiftedConc = np.hstack((psiLiftedConc, psiLifted))
print('++++++++++++++++++++++++++++++++++++++++++++++++++')
print('The global reduction basis could have', np.shape(psiLiftedConc)[1], '"mode(s)".')
# perform SVD
startTime = time.time()
U, S, Vt = np.linalg.svd(psiLiftedConc, full_matrices=False)
endTime = time.time()
timeGlobalReducedList[2] += endTime - startTime
if energyTol is not None and nGlobalModes is None:
    energy = np.cumsum(S**2) / np.sum(S**2)
    r = np.searchsorted(energy, energyTol) + 1
elif nGlobalModes is not None and energyTol is None:
    r = nGlobalModes
else:
    raise ValueError('Specify either nGlobalModes or energyTol to get number of global modes')
print('The global reduction basis has', r, '"mode(s)".')
psiLiftedGlobal = U[:, :r] # now it is not just concatenated but also corrected
print('The shape of the global reduction basis is:', np.shape(psiLiftedGlobal))

#%% get the mode bases for BC configurations to be investigated
configurationsDict = GetUnconstrainedEigenmodesForConstrainedNodeNumbersList(K=K, M=M, constrainedNodeNumbersList=constrainedNodeNumbersConfigurationsList,
                                                                             nEigenmodes=nEigenmodesForSpecificReduction)    
timeSpecReducedList[0] += configurationsDict['time']
psiBCConfigurationsList = configurationsDict['unconstrainedEigenmodesList'] # psiBC
freeDOFSConfigurationsList = configurationsDict['freeDOFSList'] # same for ref or not ref
constrainedDOFSConfigurationsList = configurationsDict['constrainedDOFSList']
KBCConfigurationsList = configurationsDict['KBCList']
MBCConfigurationsList = configurationsDict['MBCList']
# now also get the constrained entries in the eigenvectors to get psi matrix
psiLiftedConfigurationsList = []
for i, psiBCConfiguration in enumerate(psiBCConfigurationsList):
    freeDOFS = freeDOFSConfigurationsList[i]
    psiLiftedConfigurationsList += [GetConstrainedEigenmodesOutOfUnconstrainedEigenmodes(unconstrainedEigenmodes=psiBCConfiguration,
                                                                                         nDOF=nDOF, nEigenmodes=nEigenmodesForSpecificReduction,
                                                                                         freeDOFS=freeDOFS)]
# ... and postprocess this matrices; take the first corrected eigenmode of the snapshots (first snapshot) as reference
psiLiftedConfigurationsList = PostprocessEigenmodesList(psiList=psiLiftedSnapshotsList[0:1]+psiLiftedConfigurationsList, M=M, nEigenmodes=nEigenmodesForSpecificReduction)[1:] # only the last ones are needed
if nConfigurations < 6:
    if nEigenmodesForSpecificReduction >= 3:
        nEigenmodesToPlot = 3
    else:
        nEigenmodesToPlot = nEigenmodesForSpecificReduction
    PlotEigenmodes(psiList=psiLiftedConfigurationsList,
                   constrainedNodeNumbersList=constrainedNodeNumbersConfigurationsList,
                   nodeNumbersList=nodeNumbersList, nEigenmodesToPlot=nEigenmodesToPlot, figsize=[8, 3])

#%% interpolate between the snapshots to get the eigenmodes at the configurations
psiLiftedSnapshot0 = psiLiftedSnapshotsList[0]
gammaSnapshotsList = []
for psiLiftedSnapshot in psiLiftedSnapshotsList:
    # project the bases into a tangent space using a helping matrix A
    startTime = time.time()
    A = (np.eye(nDOF)-psiLiftedSnapshot0@psiLiftedSnapshot0.T) @ psiLiftedSnapshot @ np.linalg.inv(psiLiftedSnapshot0.T@psiLiftedSnapshot)
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    gammaSnapshotsList += [U @ np.arctan(np.diag(S)) @ Vt]
    endTime = time.time()
    timeIntReducedList[3] += endTime - startTime
# now interpolate between the snapshots to get configuration; standard interpolation in tangent space
gammaConfigurationsList = []
print('++++++++++++++++++++++++++++++++++++++++++++++++++')
for constrainedNodeNumberConfiguration in constrainedNodeNumbersConfigurationsList:
    # for each constrained node, get previous and following snapshot
    snapshotNumberPrevious = np.searchsorted(constrainedNodeNumbersSnapshotsList, constrainedNodeNumberConfiguration) - 1
    if snapshotNumberPrevious < 0:
        snapshotNumberPrevious = 0
    parameterPrevious = constrainedNodeNumbersSnapshotsList[snapshotNumberPrevious]
    parameterFollowing = constrainedNodeNumbersSnapshotsList[snapshotNumberPrevious+1]
    print('Interpolation for constrained node number', str(constrainedNodeNumberConfiguration) + ':', '\nprev. snapshot node:', parameterPrevious, '\nfollowing snapshot node:', parameterFollowing)
    startTime = time.time()
    gammaConfiguration = StandardInterpolationInTangentSpace(gammaSnapshotsList[snapshotNumberPrevious], gammaSnapshotsList[snapshotNumberPrevious+1],
                                                             parameterPrevious, parameterFollowing, constrainedNodeNumberConfiguration)
    endTime = time.time()
    timeIntReducedList[3] += endTime - startTime
    gammaConfigurationsList += [gammaConfiguration]
# and now map back to get the interpolated modes for the configurations
psiLiftedInterpolatedConfigurationsList = []
for gammaConfiguration in gammaConfigurationsList:
    startTime = time.time()
    U, S, Vt = np.linalg.svd(gammaConfiguration, full_matrices=False)
    psiLiftedInterpolatedConfiguration = psiLiftedSnapshot0 @ Vt.T @ np.cos(np.diag(S)) + U @ np.sin(np.diag(S))
    endTime = time.time()
    timeIntReducedList[3] += endTime - startTime
    psiLiftedInterpolatedConfigurationsList += [psiLiftedInterpolatedConfiguration]
if nConfigurations < 6:
    if nEigenmodesInSnapshots >= 3:
        nEigenmodesToPlot = 3
    else:
        nEigenmodesToPlot = nEigenmodesInSnapshots
    PlotEigenmodes(psiList=psiLiftedInterpolatedConfigurationsList,
                   constrainedNodeNumbersList=constrainedNodeNumbersConfigurationsList,
                   nodeNumbersList=nodeNumbersList, nEigenmodesToPlot=nEigenmodesToPlot, figsize=[8, 3])

#%% (1) solve the static equation for displacement without reduction
displacementNonReducedList = []
for i in range(nConfigurations):
    freeDOFSConfiguration = freeDOFSConfigurationsList[i] # same for non-reduced, spec. reduced, and global reduced
    KBCConfiguration = KBCConfigurationsList[i] # same for non reduced, spec. reduced, and global reduced
    MBCConfiguration = MBCConfigurationsList[i] # same for non reduced, spec. reduced, and global reduced
    # do not reduce the system
    fBCConfiguration = [0] * len(freeDOFSConfiguration)
    displacementNonReduced = []
    for f in fList:
        fBCConfiguration[-2] = f # force acts on displacement of last node; # same for non reduced, spec. reduced, and global reduced
        startTime = time.time()
        uBCNonReduced = solve(KBCConfiguration, fBCConfiguration)
        endTime = time.time()
        timeNonReducedList[1] += endTime - startTime  # add this to non reduced time
        # now get the full displacement vector
        uLiftedNonReduced = GetConstrainedEigenmodesOutOfUnconstrainedEigenmodes(unconstrainedEigenmodes=uBCNonReduced,
                                                                                 freeDOFS=freeDOFSConfiguration,
                                                                                 nDOF=nDOF, nEigenmodes=1) # set this to 1 here such that it works for 1D displacement
        displacementNonReduced += [uLiftedNonReduced]
    displacementNonReducedList += [displacementNonReduced]

#%% (2) solve the static equation for displacement using a specific reduction base each time
displacementSpecReducedList = []
for i, psiBCConfiguration in enumerate(psiBCConfigurationsList):
    freeDOFSConfiguration = freeDOFSConfigurationsList[i] # same for non-reduced, spec. reduced, and global reduced
    KBCConfiguration = KBCConfigurationsList[i] # same for non reduced, spec. reduced, and global reduced
    MBCConfiguration = MBCConfigurationsList[i] # same for non reduced, spec. reduced, and global reduced
    # reduce the system
    KBCSpecReduced = psiBCConfiguration.T @ KBCConfiguration @ psiBCConfiguration
    fBCConfiguration = [0] * len(freeDOFSConfiguration)
    displacementSpecReduced = []
    for f in fList:
        fBCConfiguration[-2] = f # force acts on displacement of last node
        fBCSpecReduced = psiBCConfiguration.T @ fBCConfiguration
        startTime = time.time()
        qBCSpecReduced = solve(KBCSpecReduced, fBCSpecReduced)
        endTime = time.time()
        timeSpecReducedList[1] += endTime - startTime
        uBCSpecReduced = psiBCConfiguration @ qBCSpecReduced
        # now get the full displacement vector
        uLiftedSpecReduced = GetConstrainedEigenmodesOutOfUnconstrainedEigenmodes(unconstrainedEigenmodes=uBCSpecReduced,
                                                                                  freeDOFS=freeDOFSConfiguration,
                                                                                  nDOF=nDOF, nEigenmodes=1) # set this to 1 here such that it works for 1D displacement
        displacementSpecReduced += [uLiftedSpecReduced]
    displacementSpecReducedList += [displacementSpecReduced]
    
#%% (3) solve the static equation for displacement using the interpolated reduction bases
displacementInterpolatedReducedList = []
for i, psiLiftedInterpolationConfiguration in enumerate(psiLiftedInterpolatedConfigurationsList):
    freeDOFSConfiguration = freeDOFSConfigurationsList[i] # same for non-reduced, spec. reduced, and global reduced
    KBCConfiguration = KBCConfigurationsList[i] # same for non reduced, spec. reduced, and global reduced
    MBCConfiguration = MBCConfigurationsList[i] # same for non reduced, spec. reduced, and global reduced
    # now cut the constrained DOFs out of the global reduction basis
    psiBCInterpolated = psiLiftedInterpolationConfiguration[freeDOFSConfiguration, :]
    # reduce the system
    KBCInterpolatedReduced = psiBCInterpolated.T @ KBCConfiguration @ psiBCInterpolated
    fBCConfiguration = [0] * len(freeDOFSConfiguration)
    displacementInterpolatedReduced = []
    for f in fList:
        fBCConfiguration[-2] = f # force acts on displacement of last node
        fBCInterpolatedReduced = psiBCInterpolated.T @ fBCConfiguration
        startTime = time.time()
        qBCInterpolatedReduced = solve(KBCInterpolatedReduced, fBCInterpolatedReduced)
        endTime = time.time()
        timeIntReducedList[1] += endTime - startTime
        uBCInterpolatedReduced = psiBCInterpolated @ qBCInterpolatedReduced
        # now get the full displacement vector
        uLiftedInterpolatedReduced = GetConstrainedEigenmodesOutOfUnconstrainedEigenmodes(unconstrainedEigenmodes=uBCInterpolatedReduced,
                                                                                          freeDOFS=freeDOFSConfiguration,
                                                                                          nDOF=nDOF, nEigenmodes=1) # set this to 1 here such that it works for 1D displacement
        displacementInterpolatedReduced += [uLiftedInterpolatedReduced]
    displacementInterpolatedReducedList += [displacementInterpolatedReduced]

#%% (4) solve the static equation for displacement using a global reduction base
displacementGlobalReducedList = []
for i in range(nConfigurations):
    freeDOFSConfiguration = freeDOFSConfigurationsList[i] # same for non-reduced, spec. reduced, and global reduced
    KBCConfiguration = KBCConfigurationsList[i] # same for non reduced, spec. reduced, and global reduced
    MBCConfiguration = MBCConfigurationsList[i] # same for non reduced, spec. reduced, and global reduced
    # now cut the constrained DOFs out of the global reduction basis
    psiBCGlobal = psiLiftedGlobal[freeDOFSConfiguration, :]
    # reduce the system
    KBCGlobalReduced = psiBCGlobal.T @ KBCConfiguration @ psiBCGlobal
    fBCConfiguration = [0] * len(freeDOFSConfiguration)
    displacementGlobalReduced = []
    for f in fList:
        fBCConfiguration[-2] = f # force acts on displacement of last node
        fBCGlobalReduced = psiBCGlobal.T @ fBCConfiguration
        startTime = time.time()
        qBCGlobalReduced = solve(KBCGlobalReduced, fBCGlobalReduced)
        endTime = time.time()
        timeGlobalReducedList[1] += endTime - startTime
        uBCGlobalReduced = psiBCGlobal @ qBCGlobalReduced
        # now get the full displacement vector
        uLiftedGlobalReduced = GetConstrainedEigenmodesOutOfUnconstrainedEigenmodes(unconstrainedEigenmodes=uBCGlobalReduced,
                                                                                    freeDOFS=freeDOFSConfiguration,
                                                                                    nDOF=nDOF, nEigenmodes=1) # set this to 1 here such that it works for 1D displacement
        displacementGlobalReduced += [uLiftedGlobalReduced]
    displacementGlobalReducedList += [displacementGlobalReduced]
    
#%% (5) solve the static equation for displacement using a random reduction base
psiLiftedRandom = np.random.rand(np.shape(psiLiftedGlobal)[0], np.shape(psiLiftedGlobal)[1])
displacementRandomReducedList = []
for i in range(nConfigurations):
    freeDOFSConfiguration = freeDOFSConfigurationsList[i] # same for non-reduced, spec. reduced, and global reduced
    KBCConfiguration = KBCConfigurationsList[i] # same for non reduced, spec. reduced, and global reduced
    MBCConfiguration = MBCConfigurationsList[i] # same for non reduced, spec. reduced, and global reduced
    # now cut the constrained DOFs out of the global reduction basis
    psiBCRandom = psiLiftedRandom[freeDOFSConfiguration, :]
    # reduce the system
    KBCRandomReduced = psiBCRandom.T @ KBCConfiguration @ psiBCRandom
    fBCConfiguration = [0] * len(freeDOFSConfiguration)
    displacementRandomReduced = []
    for f in fList:
        fBCConfiguration[-2] = f # force acts on displacement of last node
        fBCRandomReduced = psiBCRandom.T @ fBCConfiguration
        startTime = time.time()
        qBCRandomReduced = solve(KBCRandomReduced, fBCRandomReduced)
        endTime = time.time()
        timeRandomReducedList[1] += endTime - startTime
        uBCRandomReduced = psiBCRandom @ qBCRandomReduced
        # now get the full displacement vector
        uLiftedRandomReduced = GetConstrainedEigenmodesOutOfUnconstrainedEigenmodes(unconstrainedEigenmodes=uBCRandomReduced,
                                                                                    freeDOFS=freeDOFSConfiguration,
                                                                                    nDOF=nDOF, nEigenmodes=1) # set this to 1 here such that it works for 1D displacement
        displacementRandomReduced += [uLiftedRandomReduced]
    displacementRandomReducedList += [displacementRandomReduced]

#%% plot the displacements
if nConfigurations < 6:
    PlotDisplacement(displacementNonReducedList=displacementNonReducedList,
                     displacementSpecReducedList=displacementSpecReducedList,
                     displacementGlobalReducedList=displacementGlobalReducedList,
                     fList=fList, constrainedNodeNumbersList=constrainedNodeNumbersConfigurationsList,
                     nodeNumbersList=nodeNumbersList, figsize=[5, 4])
elif playVideos:
    # make an animation if more interpolation points are required
    PlotDisplacement(displacementNonReducedList=displacementNonReducedList,
                     displacementSpecReducedList=displacementSpecReducedList,
                     displacementGlobalReducedList=displacementGlobalReducedList,
                     fList=fList, constrainedNodeNumbersList=constrainedNodeNumbersConfigurationsList,
                     nodeNumbersList=nodeNumbersList, figsize=[5, 4], animate=True, showLegend=False, saveFigures=saveFigures)

#%% compute the errors between the results of no reduction, spec. reduction, global reduction, and random reduction
relDisplacementErrorNonReduced = 0 # stays zero
relDisplacementErrorSpecReduced = 0
relDisplacementErrorIntReduced = 0
relDisplacementErrorGlobalReduced = 0
relDisplacementErrorRandomReduced = 0
for i in range(nConfigurations):
    for j in range(nForces):
        displacementNonReduced = displacementNonReducedList[i][j][::2]
        displacementNonReducedNorm = np.linalg.norm(displacementNonReduced)
        if displacementNonReducedNorm == 0:
            continue
        displacementSpecReduced = displacementSpecReducedList[i][j][::2]
        displacementIntReduced = displacementInterpolatedReducedList[i][j][::2]
        displacementGlobalReduced = displacementGlobalReducedList[i][j][::2]
        displacementRandomReduced = displacementRandomReducedList[i][j][::2]
        relDisplacementErrorSpecReduced += np.linalg.norm(displacementSpecReduced-displacementNonReduced) / displacementNonReducedNorm
        relDisplacementErrorIntReduced += np.linalg.norm(displacementIntReduced-displacementNonReduced) / displacementNonReducedNorm
        relDisplacementErrorGlobalReduced += np.linalg.norm(displacementGlobalReduced-displacementNonReduced) / displacementNonReducedNorm
        relDisplacementErrorRandomReduced += np.linalg.norm(displacementRandomReduced-displacementNonReduced) / displacementNonReducedNorm
nTotalTests = nConfigurations * nForces
relErrorsList = [relDisplacementErrorNonReduced/nTotalTests, relDisplacementErrorSpecReduced/nTotalTests,
                 relDisplacementErrorIntReduced/nTotalTests,
                 relDisplacementErrorGlobalReduced/nTotalTests, relDisplacementErrorRandomReduced/nTotalTests]
relErrorsPercentList = np.array(relErrorsList) * 100

#%% plot time and error
timeNames = ['non reduced', 'spec. reduced', 'int. reduced (PMOR)', 'global reduced (PMOR)', 'global reduced (random)']
timesList = np.array([timeNonReducedList, timeSpecReducedList, timeIntReducedList, timeGlobalReducedList, timeRandomReducedList])
figTime, axTime = plt.subplots(1, 1, figsize=[12, 4])
bottom = np.zeros(len(timesList))
labels = ['solv. eigv.-prob.', 'solve equations', 'comp. global basis', 'interp. modes']
for i in range(timesList.shape[1]):
    axTime.bar(timeNames, timesList[:, i], bottom=bottom, label=labels[i])
    bottom += timesList[:, i]
axTime.set_ylabel('computation time in s')
axTime.grid()
axTime.legend(loc='upper left')
axError = axTime.twinx()
axError.plot(timeNames[1:], relErrorsPercentList[1:], 'o-', color='r', markersize=8)
axError.set_ylabel('rel. average euclidean distance per test in %', color='r')
axError.tick_params(axis='y', colors='r')
axError.spines['right'].set_color('r')
axError.set_yscale('log')
for i, value in enumerate(relErrorsPercentList[1:]):
    axError.text(
        timeNames[i+1], value*1.1, f'{value:.2f}%',
        color='k', ha='center', va='bottom',
        bbox=dict(facecolor='white', edgecolor='red', boxstyle='round,pad=0.3')
    )
axTime.set_ylim(top=bottom.max() * 1.1)  # leave 10% space above the tallest bar
axError.set_ylim(top=relErrorsPercentList.max() + 100)
plt.tight_layout()
plt.show()

#%% solve dynamics with global reduction basis, at first for one BC configuration

alpha = 0
beta = 1e-3
T = 1 # simulation time
dt = 1e-3
# maxStepSizeRadau = None
t = np.arange(0, T, dt)
solveNonReduced = False
integrationMethod = 'Radau'
loadMode = 'constant'

#++++++++++++++++++++++++++++++++++++++++++++++++++
# automatically compute parameters
nTimeSteps = len(t)

def PlotDynamicDisplacement(dynamicSolutionNonReduced=None, dynamicSolutionSpecReduced=None, dynamicSolutionGlobalReduced=None,
                            f=[], t=[], figsize=[8, 6]):
    SetMatplotlibSettings()
    colors = getStandardColorTemplate()
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    if dynamicSolutionNonReduced is not None:
        displacementNonReducedLastNode = dynamicSolutionNonReduced[-2, :]
        ax.plot(t, displacementNonReducedLastNode,
                color=colors[0],
                label='dynamics non reduced')
    if dynamicSolutionSpecReduced is not None:
        displacementSpecReducedLastNode = dynamicSolutionSpecReduced[-2, :]
        ax.plot(t, displacementSpecReducedLastNode,
                marker='*',
                markevery=markevery,
                color=colors[1],
                label='dynamics spec. reduced')
    if dynamicSolutionGlobalReduced is not None:
        displacementGlobalReducedLastNode = dynamicSolutionGlobalReduced[-2, :]
        ax.plot(t, displacementGlobalReducedLastNode,
                marker='p',
                markevery=markevery,
                color=colors[2],
                label='dynamics global reduced')
    ax.set_xlabel('time in s')
    ax.set_ylabel('displacement of last node in m')
    ax.legend()
    ax.grid()
    ax1 = ax.twinx()
    ax1.plot(t, f, color='red')
    ax1.set_ylabel('force in N', color='red')
    ax1.tick_params(axis='y', colors='red')
    ax1.set_ylim([-50, 50])
    plt.tight_layout()

def AnimateDynamicDisplacement(dynamicSolutionNonReduced=None, dynamicSolutionSpecReduced=None, dynamicSolutionGlobalReduced=None,
                               t=[], dynamicallyConstrainedNodeNumber=[], nodeNumbersList=[], figsize=[8, 6],
                               pauseTime=0.00001, frameInterval=1, saveFigures=False, saveString=''):
    SetMatplotlibSettings()
    colors = getStandardColorTemplate()
    if dynamicSolutionNonReduced is not None:
        nTimeSteps = np.shape(dynamicSolutionNonReduced)[1]
    elif dynamicSolutionSpecReduced is not None:
        nTimeSteps = np.shape(dynamicSolutionSpecReduced)[1]
    elif dynamicSolutionGlobalReduced is not None:
        nTimeSteps = np.shape(dynamicSolutionGlobalReduced)[1]
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    for i in range(0, nTimeSteps, frameInterval):
        ax.clear()
        if dynamicSolutionNonReduced is not None:
            ax.plot(nodeNumbersList,
                    dynamicSolutionNonReduced[::2, i],
                    color=colors[0],
                    label='dynamics non reduced')
        if dynamicSolutionSpecReduced is not None:
            ax.plot(nodeNumbersList,
                    dynamicSolutionSpecReduced[::2, i],
                    marker='*',
                    markevery=markevery,
                    color=colors[1],
                    label='dynamics spec. reduced')
        if dynamicSolutionGlobalReduced is not None:
            ax.plot(nodeNumbersList,
                    dynamicSolutionGlobalReduced[::2, i],
                    marker='p',
                    markevery=markevery,
                    color=colors[2],
                    label='dynamics global reduced')
        ax.plot(0, 0, 'o', color='k')
        ax.plot(dynamicallyConstrainedNodeNumber[i], 0, 'o', color='k')
        ax.grid()
        ax.set_ylim([-3e-5, 3e-5])
        ax.set_xlabel('node number')
        ax.set_ylabel('displacement in m')
        ax.legend()
        ax.text(0.1, 0.1, f'time = {t[i]:.2f}s',
                transform=ax.transAxes,
                fontsize=12,
                color='black',
                ha='left',
                va='bottom',
                bbox=dict(boxstyle='round,pad=0.3',
                          facecolor='white',
                          edgecolor='black',
                          linewidth=1))
        plt.tight_layout()
        if saveFigures:
            plt.savefig('images/frame' + str(i) + saveString + '.png')
        plt.pause(pauseTime)
    plt.show()
    plt.close()

# define the load
def DynamicLoad(t, reductionBasis, freeDOFSConfiguration, loadMode='sawtooth'):
    P = T/2
    amplitudeOfForce = 30
    if loadMode == 'sine':
        f = amplitudeOfForce * np.sin(2*np.pi*(1/P)*t)
    elif loadMode == 'sawtooth':
        f = amplitudeOfForce * (2 * (t/P - np.floor(t/P + 0.5)))
    elif loadMode == 'null':
        f = 0
    elif loadMode == 'constant':
        f = - 30 / 10000
    if type(freeDOFSConfiguration) is list or type(freeDOFSConfiguration) is np.ndarray:
        nDOF = len(freeDOFSConfiguration)
    else:
        nDOF = freeDOFSConfiguration
    fBC = np.zeros(nDOF)
    fBC[-2] = f # force acts on last displacement node
    if reductionBasis is not None:
        fBCGlobalReduced = reductionBasis.T @ fBC
        return fBCGlobalReduced
    else:
        return fBC
    
# define system of ODE1
def SystemOfODE1(t, x, A, B, reductionBasis, freeDOFSConfiguration, loadMode):
    dx = A @ x + B @ DynamicLoad(t, reductionBasis, freeDOFSConfiguration, loadMode=loadMode)
    return dx

print('++++++++++++++++++++++++++++++++++++++++++++++++++')
if nConfigurations < 10:
    print('Will do dynamic evaluation for specific BC configuration(s)')
    print('Will not try to move BC')
elif nConfigurations >= 10:
    print('Will try to move BC')

#%% (1) solve dynamics without reduction
if nConfigurations < 10:
    displacementDynNonReducedList = []
    if solveNonReduced:
        for i in range(nConfigurations):
            freeDOFSConfiguration = freeDOFSConfigurationsList[i] # same for non-reduced, spec. reduced, and global reduced
            KBCConfiguration = KBCConfigurationsList[i] # same for non reduced, spec. reduced, and global reduced
            MBCConfiguration = MBCConfigurationsList[i] # same for non reduced, spec. reduced, and global reduced
            # also add some Rayleigh damping
            CBCConfiguration = alpha * MBCConfiguration + beta * KBCConfiguration
            MBCNonReducedInv = np.linalg.inv(MBCConfiguration)
            # assemble A and B matrices to have first order ODE system (system is 2*numberOfReducedCoordinates big)
            A = np.block([[np.zeros([len(freeDOFSConfiguration), len(freeDOFSConfiguration)]), np.eye(len(freeDOFSConfiguration))],
                          [-MBCNonReducedInv @ KBCConfiguration, -MBCNonReducedInv @ CBCConfiguration]])
            B = np.block([[np.zeros([len(freeDOFSConfiguration), len(freeDOFSConfiguration)])],
                          [MBCNonReducedInv]])
            # dynamically simulate the system
            sBCDynNonReduced0 = np.zeros(2*len(freeDOFSConfiguration))
            qLiftedDynNonReduced = np.zeros([nDOF, nTimeSteps])
            qLiftedDynNonReduced_t = np.zeros([nDOF, nTimeSteps])
            # simulate
            print('Starting dynamic simulation with NO REDUCTION')
            startTime = time.time()
            sol = solve_ivp(SystemOfODE1,
                            (t[0], t[-1]),
                            sBCDynNonReduced0,
                            args=(A, B, None, freeDOFSConfiguration, loadMode),
                            t_eval=t,
                            # max_step=maxStepSizeRadau,
                            method=integrationMethod,
                            rtol=1e-3,
                            atol=1e-6)
            print('Dynamic simulation finished after ' + str(round(time.time()-startTime, 3)) + 's')
            # post process
            for j in range(nTimeSteps):
                sBCDynNonReduced = sol.y[:, j]
                qBCDynNonReduced = sBCDynNonReduced[0:len(freeDOFSConfiguration)]
                qBCDynNonReduced_t = sBCDynNonReduced[len(freeDOFSConfiguration):]
                qLiftedDynNonReduced[:, j] = GetConstrainedEigenmodesOutOfUnconstrainedEigenmodes(unconstrainedEigenmodes=qBCDynNonReduced,
                                                                                                     freeDOFS=freeDOFSConfiguration,
                                                                                                     nDOF=nDOF, nEigenmodes=1)[:, 0] # set this to 1 here such that it works for 1D displacement
                qLiftedDynNonReduced_t[:, j] = GetConstrainedEigenmodesOutOfUnconstrainedEigenmodes(unconstrainedEigenmodes=qBCDynNonReduced_t,
                                                                                                       freeDOFS=freeDOFSConfiguration,
                                                                                                       nDOF=nDOF, nEigenmodes=1)[:, 0] # set this to 1 here such that it works for 1D displacement
            displacementDynNonReducedList += [qLiftedDynNonReduced]

#%% (2) solve dynamics for configurations using a specific reduction base each time
if nConfigurations < 10:
    displacementDynSpecReducedList = []
    for i, psiBCConfiguration in enumerate(psiBCConfigurationsList):
        freeDOFSConfiguration = freeDOFSConfigurationsList[i] # same for non-reduced, spec. reduced, and global reduced
        KBCConfiguration = KBCConfigurationsList[i] # same for non reduced, spec. reduced, and global reduced
        MBCConfiguration = MBCConfigurationsList[i] # same for non reduced, spec. reduced, and global reduced
        # also add some Rayleigh damping
        CBCConfiguration = alpha * MBCConfiguration + beta * KBCConfiguration
        # reduce system
        MBCSpecReduced = psiBCConfiguration.T @ MBCConfiguration @ psiBCConfiguration
        KBCSpecReduced = psiBCConfiguration.T @ KBCConfiguration @ psiBCConfiguration
        CBCSpecReduced = psiBCConfiguration.T @ CBCConfiguration @ psiBCConfiguration
        MBCSpecReducedInv = np.linalg.inv(MBCSpecReduced)
        # assemble A and B matrices to have first order ODE system (system is 2*numberOfReducedCoordinates big)
        A = np.block([[np.zeros([nEigenmodesForSpecificReduction, nEigenmodesForSpecificReduction]), np.eye(nEigenmodesForSpecificReduction)],
                      [-MBCSpecReducedInv @ KBCSpecReduced, -MBCSpecReducedInv @ CBCSpecReduced]])
        B = np.block([[np.zeros([nEigenmodesForSpecificReduction, nEigenmodesForSpecificReduction])],
                      [MBCSpecReducedInv]])
        # dynamically simulate the system
        sBCDynSpecReduced0 = np.zeros(2*nEigenmodesForSpecificReduction)
        qLiftedDynSpecReduced = np.zeros([nDOF, nTimeSteps])
        qLiftedDynSpecReduced_t = np.zeros([nDOF, nTimeSteps])
        # simulate
        print('Starting dynamic simulation with SPECIFIC REDUCTION')
        startTime = time.time()
        sol = solve_ivp(SystemOfODE1,
                        (t[0], t[-1]),
                        sBCDynSpecReduced0,
                        args=(A, B, psiBCConfiguration, freeDOFSConfiguration, loadMode),
                        t_eval=t,
                        # max_step=maxStepSizeRadau,
                        method=integrationMethod,
                        rtol=1e-3,
                        atol=1e-6)
        print('Dynamic simulation finished after ' + str(round(time.time()-startTime, 3)) + 's')
        # post process
        for j in range(nTimeSteps):
            sBCDynSpecReduced = sol.y[:, j]
            qBCDynSpecReduced = psiBCConfiguration @ sBCDynSpecReduced[0:nEigenmodesForSpecificReduction]
            qBCDynSpecReduced_t = psiBCConfiguration @ sBCDynSpecReduced[nEigenmodesForSpecificReduction:]
            qLiftedDynSpecReduced[:, j] = GetConstrainedEigenmodesOutOfUnconstrainedEigenmodes(unconstrainedEigenmodes=qBCDynSpecReduced,
                                                                                                 freeDOFS=freeDOFSConfiguration,
                                                                                                 nDOF=nDOF, nEigenmodes=1)[:, 0] # set this to 1 here such that it works for 1D displacement
            qLiftedDynSpecReduced_t[:, j] = GetConstrainedEigenmodesOutOfUnconstrainedEigenmodes(unconstrainedEigenmodes=qBCDynSpecReduced_t,
                                                                                                   freeDOFS=freeDOFSConfiguration,
                                                                                                   nDOF=nDOF, nEigenmodes=1)[:, 0] # set this to 1 here such that it works for 1D displacement
        displacementDynSpecReducedList += [qLiftedDynSpecReduced]

#%% (3) solve dynamics for configurations using a global reduction base each time
if nConfigurations < 10:
    displacementDynGlobalReducedList = []
    for i in range(nConfigurations):
        freeDOFSConfiguration = freeDOFSConfigurationsList[i] # same for non-reduced, spec. reduced, and global reduced
        KBCConfiguration = KBCConfigurationsList[i] # same for non reduced, spec. reduced, and global reduced
        MBCConfiguration = MBCConfigurationsList[i] # same for non reduced, spec. reduced, and global reduced
        # also add some Rayleigh damping
        CBCConfiguration = alpha * MBCConfiguration + beta * KBCConfiguration
        # reduce system
        psiBCGlobal = psiLiftedGlobal[freeDOFSConfiguration, :]
        MBCGlobalReduced = psiBCGlobal.T @ MBCConfiguration @ psiBCGlobal
        KBCGlobalReduced = psiBCGlobal.T @ KBCConfiguration @ psiBCGlobal
        CBCGlobalReduced = psiBCGlobal.T @ CBCConfiguration @ psiBCGlobal
        MBCGlobalReducedInv = np.linalg.inv(MBCGlobalReduced)
        # assemble A and B matrices to have first order ODE system (system is 2*numberOfReducedCoordinates big)
        A = np.block([[np.zeros([nGlobalModes, nGlobalModes]), np.eye(nGlobalModes)],
                      [-MBCGlobalReducedInv @ KBCGlobalReduced, -MBCGlobalReducedInv @ CBCGlobalReduced]])
        B = np.block([[np.zeros([nGlobalModes, nGlobalModes])],
                      [MBCGlobalReducedInv]])
        # dynamically simulate the system
        sBCDynGlobalReduced0 = np.zeros(2*nGlobalModes)
        qLiftedDynGlobalReduced = np.zeros([nDOF, nTimeSteps])
        qLiftedDynGlobalReduced_t = np.zeros([nDOF, nTimeSteps])
        # simulate
        print('Starting dynamic simulation with GLOBAL REDUCTION')
        startTime = time.time()
        sol = solve_ivp(SystemOfODE1,
                        (t[0], t[-1]),
                        sBCDynGlobalReduced0,
                        args=(A, B, psiBCGlobal, freeDOFSConfiguration, loadMode),
                        t_eval=t,
                        # max_step=maxStepSizeRadau,
                        method=integrationMethod,
                        rtol=1e-3,
                        atol=1e-6)
        print('Dynamic simulation finished after ' + str(round(time.time()-startTime, 3)) + 's')
        # post process
        for j in range(nTimeSteps):
            sBCDynGlobalReduced = sol.y[:, j]
            qBCDynGlobalReduced = psiBCGlobal @ sBCDynGlobalReduced[0:nGlobalModes]
            qBCDynGlobalReduced_t = psiBCGlobal @ sBCDynGlobalReduced[nGlobalModes:]
            qLiftedDynGlobalReduced[:, j] = GetConstrainedEigenmodesOutOfUnconstrainedEigenmodes(unconstrainedEigenmodes=qBCDynGlobalReduced,
                                                                                                 freeDOFS=freeDOFSConfiguration,
                                                                                                 nDOF=nDOF, nEigenmodes=1)[:, 0] # set this to 1 here such that it works for 1D displacement
            qLiftedDynGlobalReduced_t[:, j] = GetConstrainedEigenmodesOutOfUnconstrainedEigenmodes(unconstrainedEigenmodes=qBCDynGlobalReduced_t,
                                                                                                   freeDOFS=freeDOFSConfiguration,
                                                                                                   nDOF=nDOF, nEigenmodes=1)[:, 0] # set this to 1 here such that it works for 1D displacement
        displacementDynGlobalReducedList += [qLiftedDynGlobalReduced]

#%% plot the displacements and also compute relative errors
if nConfigurations < 10:
    relDisplacementErrorDynNonReduced = 0 # stays zero
    relDisplacementErrorDynSpecReduced = 0
    relDisplacementErrorDynGlobalReduced = 0
    for i in range(nConfigurations):
        if displacementDynNonReducedList != []:
            dynamicSolutionNonReduced = displacementDynNonReducedList[i]
        else:
            dynamicSolutionNonReduced= None
        if displacementDynSpecReducedList != []:
            dynamicSolutionSpecReduced = displacementDynSpecReducedList[i]
        else:
            dynamicSolutionSpecReduced= None
        if displacementDynGlobalReducedList != []:
            dynamicSolutionGlobalReduced = displacementDynGlobalReducedList[i]
        else:
            dynamicSolutionGlobalReduced= None
        f = []
        for j in t:
            f += [DynamicLoad(j, None, freeDOFSConfigurationsList[i], loadMode=loadMode)[-2]]
        PlotDynamicDisplacement(dynamicSolutionNonReduced=dynamicSolutionNonReduced,
                                dynamicSolutionSpecReduced=dynamicSolutionSpecReduced,
                                dynamicSolutionGlobalReduced=dynamicSolutionGlobalReduced, t=t, f=f)
        if playVideos:
            dynamicallyConstrainedNodeNumber = [constrainedNodeNumbersConfigurationsList[i]] * nTimeSteps
            AnimateDynamicDisplacement(dynamicSolutionNonReduced=dynamicSolutionNonReduced,
                                    dynamicSolutionSpecReduced=dynamicSolutionSpecReduced,
                                    dynamicSolutionGlobalReduced=dynamicSolutionGlobalReduced, t=t,
                                    dynamicallyConstrainedNodeNumber=dynamicallyConstrainedNodeNumber,
                                    nodeNumbersList=nodeNumbersList, figsize=[8, 6],
                                    pauseTime=1e-10, frameInterval=1, saveFigures=True,
                                    saveString='mode' + str(i))
        if solveNonReduced:
            # compute error
            dynamicSolutionNonReducedFlat = dynamicSolutionNonReduced.flatten()
            dynamicSolutionSpecReducedFlat = dynamicSolutionSpecReduced.flatten()
            dynamicSolutionGlobalReducedFlat = dynamicSolutionGlobalReduced.flatten()
            relDisplacementErrorDynSpecReduced += np.linalg.norm(dynamicSolutionSpecReducedFlat-dynamicSolutionNonReducedFlat) / np.linalg.norm(dynamicSolutionNonReducedFlat)
            relDisplacementErrorDynGlobalReduced += np.linalg.norm(dynamicSolutionGlobalReducedFlat-dynamicSolutionNonReducedFlat) / np.linalg.norm(dynamicSolutionNonReducedFlat)
    nTotalTests = nConfigurations
    relDynErrorsList = [relDisplacementErrorDynNonReduced/nTotalTests, relDisplacementErrorDynSpecReduced/nTotalTests,
                        relDisplacementErrorDynGlobalReduced/nTotalTests]
    relDynErrorsPercentList = np.array(relDynErrorsList) * 100

#%% plot error
if nConfigurations < 10:
    if solveNonReduced:
        figError, axError = plt.subplots(1, 1, figsize=[12, 4])
        axError.bar([timeNames[0], timeNames[1], timeNames[3]], relDynErrorsPercentList)
        axError.set_ylabel('rel. average euclidean distance per test in %')
        axError.grid()
        
#%% (1) moving BCs with different modal space each configuration
timePerConfiguration = T / nConfigurations
# set the starting condition for the system
qLiftedDynNonReduced = np.zeros([nDOF, nTimeSteps])
qLiftedDynNonReduced_t = np.zeros([nDOF, nTimeSteps])
tList = np.array_split(t, nConfigurations)
timeStepsCompleted = 0
dynamicConstrainedList = []
for i in range(nConfigurations):
    dynamicConstrainedList += [constrainedNodeNumbersConfigurationsList[i]] * len(tList[i])
    # # set up the system valid for exactly the current BCs the system is at
    freeDOFSConfiguration = freeDOFSConfigurationsList[i] # same for non-reduced, spec. reduced, and global reduced
    KBCConfiguration = KBCConfigurationsList[i] # same for non reduced, spec. reduced, and global reduced
    MBCConfiguration = MBCConfigurationsList[i] # same for non reduced, spec. reduced, and global reduced
    # # also add some Rayleigh damping
    CBCConfiguration = alpha * MBCConfiguration + beta * KBCConfiguration
    # # reduce system
    # psiBCGlobal = psiLiftedGlobal[freeDOFSConfiguration, :]
    # MBCGlobalReduced = psiBCGlobal.T @ MBCConfiguration @ psiBCGlobal
    # KBCGlobalReduced = psiBCGlobal.T @ KBCConfiguration @ psiBCGlobal
    # CBCGlobalReduced = psiBCGlobal.T @ CBCConfiguration @ psiBCGlobal
    # MBCGlobalReducedInv = np.linalg.inv(MBCGlobalReduced)
    # assemble A and B matrices to have first order ODE system (system is 2*numberOfReducedCoordinates big)
    A = np.block([[np.zeros([nDOF-2, nDOF-2]), np.eye(nDOF-2)],
                  [-np.linalg.inv(MBCConfiguration) @ KBCConfiguration, -np.linalg.inv(MBCConfiguration) @ CBCConfiguration]])
    B = np.block([[np.zeros([nDOF-2, nDOF-2])],
                  [np.linalg.inv(MBCConfiguration)]])
    # get initial values for following simulation
    if timeStepsCompleted == 0:
        offset = 0
    else:
        offset = 1
    s0BC = qLiftedDynNonReduced[freeDOFSConfiguration, timeStepsCompleted-offset]
    s0BC_t = qLiftedDynNonReduced_t[freeDOFSConfiguration, timeStepsCompleted-offset]
    # simulate
    print('Starting dynamic simulation with NON REDUCED for BC configuration', i)
    print('Simulating in the time between', tList[i][0], 'to', tList[i][-1])
    startTime = time.time()
    sol = solve_ivp(SystemOfODE1,
                    (tList[i][0], tList[i][-1]),
                    np.hstack((s0BC, s0BC_t)),
                    args=(A, B, None, freeDOFSConfiguration, loadMode),
                    t_eval=tList[i],
                    # max_step=maxStepSizeRadau,
                    method=integrationMethod,
                    rtol=1e-3,
                    atol=1e-6)
    for j in range(len(tList[i])):
        sBCNonReduced = sol.y[:, j]
        qBCNonReduced = sBCNonReduced[0:nDOF-2]
        qBCNonReduced_t = sBCNonReduced[nDOF-2:]
        qLiftedDynNonReduced[:, timeStepsCompleted] = GetConstrainedEigenmodesOutOfUnconstrainedEigenmodes(unconstrainedEigenmodes=qBCNonReduced,
                                                                                                              freeDOFS=freeDOFSConfiguration,
                                                                                                              nDOF=nDOF, nEigenmodes=1)[:, 0] # set this to 1 here such that it works for 1D displacement
        qLiftedDynNonReduced_t[:, timeStepsCompleted] = GetConstrainedEigenmodesOutOfUnconstrainedEigenmodes(unconstrainedEigenmodes=qBCNonReduced_t,
                                                                                                                freeDOFS=freeDOFSConfiguration,
                                                                                                                nDOF=nDOF, nEigenmodes=1)[:, 0] # set this to 1 here such that it works for 1D displacement
        timeStepsCompleted += 1




#%%
timePerConfiguration = T / nConfigurations
# set the starting condition for the system
qLiftedDynGlobalReduced = np.zeros([nDOF, nTimeSteps])
qLiftedDynGlobalReduced_t = np.zeros([nDOF, nTimeSteps])
tList = np.array_split(t, nConfigurations)
timeStepsCompleted = 0
dynamicConstrainedList = []
for i in range(nConfigurations):
    dynamicConstrainedList += [constrainedNodeNumbersConfigurationsList[i]] * len(tList[i])
    # set up the system valid for exactly the current BCs the system is at
    freeDOFSConfiguration = freeDOFSConfigurationsList[i] # same for non-reduced, spec. reduced, and global reduced
    KBCConfiguration = KBCConfigurationsList[i] # same for non reduced, spec. reduced, and global reduced
    MBCConfiguration = MBCConfigurationsList[i] # same for non reduced, spec. reduced, and global reduced
    # also add some Rayleigh damping
    CBCConfiguration = alpha * MBCConfiguration + beta * KBCConfiguration
    # reduce system
    psiBCGlobal = psiLiftedGlobal[freeDOFSConfiguration, :]
    MBCGlobalReduced = psiBCGlobal.T @ MBCConfiguration @ psiBCGlobal
    KBCGlobalReduced = psiBCGlobal.T @ KBCConfiguration @ psiBCGlobal
    CBCGlobalReduced = psiBCGlobal.T @ CBCConfiguration @ psiBCGlobal
    MBCGlobalReducedInv = np.linalg.inv(MBCGlobalReduced)
    # assemble A and B matrices to have first order ODE system (system is 2*numberOfReducedCoordinates big)
    A = np.block([[np.zeros([nGlobalModes, nGlobalModes]), np.eye(nGlobalModes)],
                  [-MBCGlobalReducedInv @ KBCGlobalReduced, -MBCGlobalReducedInv @ CBCGlobalReduced]])
    B = np.block([[np.zeros([nGlobalModes, nGlobalModes])],
                  [MBCGlobalReducedInv]])
    # get initial values for following simulation
    if timeStepsCompleted == 0:
        offset = 0
    else:
        offset = 1
    s0BC = psiBCGlobal.T @ qLiftedDynGlobalReduced[freeDOFSConfiguration, timeStepsCompleted-offset]
    s0BC_t = psiBCGlobal.T @ qLiftedDynGlobalReduced_t[freeDOFSConfiguration, timeStepsCompleted-offset]
    # simulate
    print('Starting dynamic simulation with GLOBAL REDUCTION for BC configuration', i)
    print('Simulating in the time between', tList[i][0], 'to', tList[i][-1])
    startTime = time.time()
    sol = solve_ivp(SystemOfODE1,
                    (tList[i][0], tList[i][-1]),
                    np.hstack((s0BC, s0BC_t)),
                    args=(A, B, psiBCGlobal, freeDOFSConfiguration, loadMode),
                    t_eval=tList[i],
                    # max_step=maxStepSizeRadau,
                    method=integrationMethod,
                    rtol=1e-3,
                    atol=1e-6)
    for j in range(len(tList[i])):
        sBCDynGlobalReduced = sol.y[:, j]
        qBCDynGlobalReduced = psiBCGlobal @ sBCDynGlobalReduced[0:nGlobalModes]
        qBCDynGlobalReduced_t = psiBCGlobal @ sBCDynGlobalReduced[nGlobalModes:]
        qLiftedDynGlobalReduced[:, timeStepsCompleted] = GetConstrainedEigenmodesOutOfUnconstrainedEigenmodes(unconstrainedEigenmodes=qBCDynGlobalReduced,
                                                                                                              freeDOFS=freeDOFSConfiguration,
                                                                                                              nDOF=nDOF, nEigenmodes=1)[:, 0] # set this to 1 here such that it works for 1D displacement
        qLiftedDynGlobalReduced_t[:, timeStepsCompleted] = GetConstrainedEigenmodesOutOfUnconstrainedEigenmodes(unconstrainedEigenmodes=qBCDynGlobalReduced_t,
                                                                                                                freeDOFS=freeDOFSConfiguration,
                                                                                                                nDOF=nDOF, nEigenmodes=1)[:, 0] # set this to 1 here such that it works for 1D displacement
        timeStepsCompleted += 1
# if playVideos:
AnimateDynamicDisplacement(#dynamicSolutionNonReduced=dynamicSolutionNonReduced,
                        # dynamicSolutionSpecReduced=dynamicSolutionSpecReduced,
                        dynamicSolutionGlobalReduced=qLiftedDynGlobalReduced,
                        t=t,
                        dynamicallyConstrainedNodeNumber=dynamicConstrainedList,
                        nodeNumbersList=nodeNumbersList, figsize=[8, 6],
                        pauseTime=1e-10, frameInterval=1, saveFigures=True)
    
#%% ( ) moving BCs with enforced BCs
# penaltyFactorAlpha = 1e9
# penaltyFactorBeta = 0.04 * penaltyFactorAlpha
# if nConfigurations > 10:
#     displacementDynGlobalReducedList = []
#     def GetBoundaryConfigurationWithTime(t):
#         desiredConfiguration = int(nConfigurations * (t/T))
#         if desiredConfiguration > (nConfigurations-1):
#             print('WARNNING: Simulation exceeds specified time T')
#             return nConfigurations - 1
#         return desiredConfiguration
#     def RMatrix(constrainedDOFSConfiguration, nDOF):
#         nConstrainedDOFs = len(constrainedDOFSConfiguration)
#         R = np.zeros([nConstrainedDOFs, nDOF])
#         for i, dof in enumerate(constrainedDOFSConfiguration):
#             R[i, dof] = 1
#         return R
#     # define system of ODE1 with moving BC
#     def SystemOfODE1WithMovingBCs(t, x, A, B, reductionBasis, nDOF, nGlobalModes, penaltyFactorAlpha, penaltyFactorBeta, loadMode):
#         currentBCConfiguration = GetBoundaryConfigurationWithTime(t)
#         currentConstrainedDOFS = constrainedDOFSConfigurationsList[currentBCConfiguration]
#         currentFreeDOFS = freeDOFSConfigurationsList[currentBCConfiguration]
#         currentR = RMatrix(currentConstrainedDOFS, nDOF)
#         # get the R matrix but in reduced coordinates
#         G = currentR @ reductionBasis
#         # penalty force (reduced)
#         fLiftedPenalty = - penaltyFactorAlpha * G.T @ (G @ x[:nGlobalModes]) - penaltyFactorBeta * G.T @ (G @ x[nGlobalModes:])
#         # external force (reduced)
#         fLiftedExternal = DynamicLoad(t, reductionBasis, nDOF, loadMode)
#         # state derivative
#         dx = A @ x + B @ (fLiftedExternal + fLiftedPenalty)
#         return dx
#     C = alpha * M + beta * K # does not depend on configuration
#     MLiftedGlobalReduced = psiLiftedGlobal.T @ M @ psiLiftedGlobal # does not depend on configuration
#     KLiftedGlobalReduced = psiLiftedGlobal.T @ K @ psiLiftedGlobal # does not depend on configuration
#     CLiftedGlobalReduced = psiLiftedGlobal.T @ C @ psiLiftedGlobal # does not depend on configuration
#     MLiftedGlobalReducedInv = np.linalg.inv(MLiftedGlobalReduced)
#     # assemble A and B matrices to have first order ODE system (system is 2*numberOfReducedCoordinates big)
#     A = np.block([[np.zeros([nGlobalModes, nGlobalModes]), np.eye(nGlobalModes)],
#                   [-MLiftedGlobalReducedInv @ KLiftedGlobalReduced, -MLiftedGlobalReducedInv @ CLiftedGlobalReduced]])
#     B = np.block([[np.zeros([nGlobalModes, nGlobalModes])],
#                   [MLiftedGlobalReducedInv]])
#     # dynamically simulate the system
#     sLiftedDynMovingBCGlobalReduced0 = np.zeros(2*nGlobalModes)
#     qLiftedDynMovingBCGlobalReduced = np.zeros([nDOF, nTimeSteps])
#     qLiftedDynMovingBCGlobalReduced_t = np.zeros([nDOF, nTimeSteps])
#     # simulate
#     print('Starting dynamic simulation with GLOBAL REDUCTION and MOVING BCs')
#     startTime = time.time()
#     sol = solve_ivp(SystemOfODE1WithMovingBCs,
#                     (t[0], t[-1]),
#                     sLiftedDynMovingBCGlobalReduced0,
#                     args=(A, B, psiLiftedGlobal, nDOF, nGlobalModes,
#                           penaltyFactorAlpha, penaltyFactorBeta, loadMode), # free DOFs here are nDOF of system
#                     t_eval=t,
#                     # max_step=maxStepSizeRadau,
#                     method=integrationMethod,
#                     rtol=1e-3,
#                     atol=1e-6)
#     print('Dynamic simulation finished after ' + str(round(time.time()-startTime, 3)) + 's')
#     dynamicallyConstrainedNodeNumber = []
#     dynamicSolutionGlobalReduced = np.zeros([nDOF, nTimeSteps])
#     for i, j in enumerate(t):
#         dynamicallyConstrainedNodeNumber += [constrainedNodeNumbersConfigurationsList[GetBoundaryConfigurationWithTime(j)]]
#         sLiftedDynMovingBC = sol.y[0:nGlobalModes]
#         dynamicSolutionGlobalReduced[:, i] = psiLiftedGlobal @ sLiftedDynMovingBC[:, i]
#     AnimateDynamicDisplacement(# dynamicSolutionNonReduced=dynamicSolutionNonReduced,
#                                # dynamicSolutionSpecReduced=dynamicSolutionSpecReduced,
#                                dynamicSolutionGlobalReduced=dynamicSolutionGlobalReduced, t=t,
#                                dynamicallyConstrainedNodeNumber=dynamicallyConstrainedNodeNumber,
#                                nodeNumbersList=nodeNumbersList, figsize=[8, 6],
#                                pauseTime=1e-10, frameInterval=1, saveFigures=False)

#%%

# #++++++++++++++++++++++++++++++++++++++++++++++++++
# # get the specific matrices, ... for the specific BC configuration
# freeDOFSConfiguration = freeDOFSConfigurationsList[configurationNumber] # same for non-reduced, spec. reduced, and global reduced
# KBCConfiguration = KBCConfigurationsList[configurationNumber] # same for non reduced, spec. reduced, and global reduced
# MBCConfiguration = MBCConfigurationsList[configurationNumber] # same for non reduced, spec. reduced, and global reduced
# # also add some damping
# CBCConfiguration = alpha * MBCConfiguration + beta * KBCConfiguration
# # now cut the constrained DOFs out of the global reduction basis
# psiBCGlobal = psiLiftedGlobal[freeDOFSConfiguration, :]

# #++++++++++++++++++++++++++++++++++++++++++++++++++
# # reduce matrices
# MBCGlobalReduced = psiBCGlobal.T @ MBCConfiguration @ psiBCGlobal
# KBCGlobalReduced = psiBCGlobal.T @ KBCConfiguration @ psiBCGlobal
# CBCGlobalReduced = psiBCGlobal.T @ CBCConfiguration @ psiBCGlobal
# MBCGlobalReducedInv = np.linalg.inv(MBCGlobalReduced)

# #++++++++++++++++++++++++++++++++++++++++++++++++++
# # assemble CONSTANT A and B matrices
# # since we are reducing the
# # - system of ODE1 has nGlobalModes * 2 equations
# # - ODE2 has nGlobalModes equations
# A = np.block([[np.zeros([nGlobalModes, nGlobalModes]), np.eye(nGlobalModes)],
#               [-MBCGlobalReducedInv @ KBCGlobalReduced, -MBCGlobalReducedInv @ CBCGlobalReduced]])
# B = np.block([[np.zeros([nGlobalModes, nGlobalModes])],
#               [MBCGlobalReducedInv]])


# #++++++++++++++++++++++++++++++++++++++++++++++++++
# # dynamically simulate the system
# # initialize soultion vector (row represents one state, column one timestep)
# sBCDynGlobalReduced0 = np.zeros(2*nGlobalModes)
# qLiftedDynGlobalReduced = np.zeros([nDOF, nTimeSteps])
# qLiftedDynGlobalReduced_t = np.zeros([nDOF, nTimeSteps])
# # simulate
# print('Starting dynamic simulation')
# startTime = time.time()
# sol = solve_ivp(SystemOfODE1,
#                 (t[0], t[-1]),
#                 sBCDynGlobalReduced0,
#                 t_eval=t,
#                 method='Radau',
#                 rtol=1e-3,
#                 atol=1e-6)
# print('Dynamic simulation finished')
# print('Dynamic simulation needed', str(round(time.time()-startTime, 3)) + 's')
# # post process
# for i in range(nTimeSteps):
#     sBCDynGlobalReduced = sol.y[:, i]
#     qBCDynGlobalReduced = psiBCGlobal @ sBCDynGlobalReduced[0:nGlobalModes]
#     qBCDynGlobalReduced_t = psiBCGlobal @ sBCDynGlobalReduced[nGlobalModes:]
#     qLiftedDynGlobalReduced[:, i] = GetConstrainedEigenmodesOutOfUnconstrainedEigenmodes(unconstrainedEigenmodes=qBCDynGlobalReduced,
#                                                                                          freeDOFS=freeDOFSConfiguration,
#                                                                                          nDOF=nDOF, nEigenmodes=1)[:, 0] # set this to 1 here such that it works for 1D displacement
#     qLiftedDynGlobalReduced_t[:, i] = GetConstrainedEigenmodesOutOfUnconstrainedEigenmodes(unconstrainedEigenmodes=qBCDynGlobalReduced_t,
#                                                                                            freeDOFS=freeDOFSConfiguration,
#                                                                                            nDOF=nDOF, nEigenmodes=1)[:, 0] # set this to 1 here such that it works for 1D displacement
# plt.plot(t, qLiftedDynGlobalReduced[-2, :])

# if playVideos:
#     PlotDynamicDisplacement(qLiftedDynGlobalReduced,
#                             constrainedNodeNumbersList=[constrainedNodeNumbersConfigurationsList[configurationNumber]],
#                             nodeNumbersList=nodeNumbersList, pauseTime=1e-10, frameInterval=2,
#                             t=t)





#%%





# displacementGlobalReducedList = []
# for i in range(nConfigurations):
#     freeDOFSConfiguration = freeDOFSConfigurationsList[i] # same for non-reduced, spec. reduced, and global reduced
#     KBCConfiguration = KBCConfigurationsList[i] # same for non reduced, spec. reduced, and global reduced
#     MBCConfiguration = MBCConfigurationsList[i] # same for non reduced, spec. reduced, and global reduced
#     # now cut the constrained DOFs out of the global reduction basis
#     psiBCGlobal = psiLiftedGlobal[freeDOFSConfiguration, :]
#     # reduce the system
#     KBCGlobalReduced = psiBCGlobal.T @ KBCConfiguration @ psiBCGlobal
#     fBCConfiguration = [0] * len(freeDOFSConfiguration)
#     displacementGlobalReduced = []
#     for f in fList:
#         fBCConfiguration[-2] = f # force acts on displacement of last node
#         fBCConfigurationGlobalReduced = psiBCGlobal.T @ fBCConfiguration
#         startTime = time.time()
#         qBCGlobalReduced = solve(KBCGlobalReduced, fBCConfigurationGlobalReduced)
#         endTime = time.time()
#         timeGlobalReducedList[1] += endTime - startTime
#         uBCGlobalReduced = psiBCGlobal @ qBCGlobalReduced
#         # now get the full displacement vector
#         uLiftedGlobalReduced = GetConstrainedEigenmodesOutOfUnconstrainedEigenmodes(unconstrainedEigenmodes=uBCGlobalReduced,
#                                                                                     freeDOFS=freeDOFSConfiguration,
#                                                                                     nDOF=nDOF, nEigenmodes=1) # set this to 1 here such that it works for 1D displacement
#         displacementGlobalReduced += [uLiftedGlobalReduced]
#     displacementGlobalReducedList += [displacementGlobalReduced]



#%% use the local approach to get interpolated eigenmodes
    # psiLiftedInterpolatedConfigurationsAlignedList = []
    # for i, psiLiftedInterpolatedConfiguration in enumerate(psiLiftedInterpolatedConfigurationsList):
    #     U, S, Vt = np.linalg.svd(psiLiftedConfigurationsList[i].T @ psiLiftedInterpolatedConfiguration)
    #     Q = U @ Vt
    #     psiLiftedInterpolatedConfigurationAligned = psiLiftedInterpolatedConfiguration @ Q
    #     psiLiftedInterpolatedConfigurationsAlignedList += [psiLiftedInterpolatedConfigurationAligned]
    #     principalAngles = np.arccos(np.clip(S, -1.0, 1.0))
    #     print("Principal angles (rad) between original and interpolated subspace:", principalAngles)

    # KBCInterpolationRefReduced = unconstrainedEigenmodesInterpolationRef.T @ KBCInterpolation @ unconstrainedEigenmodesInterpolationRef
    # fBCConfiguration = [0] * np.shape(MBCInterpolation)[0]
    # displacementGlobalReducedRef = []
    # for f in fList:
    #     fBCConfiguration[-2] = f # force acts on displacement of last node
    #     fBCConfigurationReduced = unconstrainedEigenmodesInterpolationRef.T @ fBCConfiguration
    #     qBCInterpolationRefReduced = solve(KBCInterpolationRefReduced, fBCConfigurationReduced)
    #     uBCInterpolationRef = unconstrainedEigenmodesInterpolationRef @ qBCInterpolationRefReduced
    #     # now get the full displacement vector
    #     uInterpolationRef = GetConstrainedEigenmodesOutOfUnconstrainedEigenmodes(unconstrainedEigenmodes=uBCInterpolationRef,
    #                                                                              freeDOFS=freeDOFSConfiguration,
    #                                                                              nDOF=nDOF, nEigenmodes=1) # set this to 1 here such that it works for 1D displacement
    #     displacementGlobalReducedRef += [uInterpolationRef]
    # displacementGlobalReducedRefList += [displacementGlobalReducedRef]

        
#%%
#     # reconstruct full deformation (with enforced BCs)
    

# psiListRefInterpolated = GetEigenmodesForConstrainedNodeNumbersList(K=K, M=M,
#                                                                     constrainedNodeNumbersList=constrainedNodeNumbersListInterpolation)
# psiListRefInterpolated = PreprocessEigenmodes(psiList=[psiListSnapshots[-2]] + psiListRefInterpolated)[1:]
# PlotEigenmodes(psiList=psiListRefInterpolated,
#                constrainedNodeNumbersList=constrainedNodeNumbersListInterpolation,
#                nodeNumbersList=nodeNumbersList, nEigenmodesToPlot=3, figsize=[8, 3])

# #%% compute displacements using the reference mode basis
# matricesDict = GetEigenmodesForConstrainedNodeNumbersList(K=K, M=M,
#                                                           getReducedMatrices=True,
#                                                           f=fList[0],
#                                                           constrainedNodeNumbersList=constrainedNodeNumbersListInterpolation)
# KWithEnforcedBC = matricesDict['KReduced']
# MWithEnforcedBC = matricesDict['MReduced']
# fWithEnforcedBC = matricesDict['fReduced']



# KRefReduced = psiListRefInterpolated[0].T @ KWithEnforcedBC @ psiListRefInterpolated[0]
# fRefReduced = psiListRefInterpolated[0].T @ fWithEnforcedBC
# qRef = solve(KRefReduced, fRefReduced)
# # reconstruct full deformation (with enforced BCs)
# u_ref = psiListRefInterpolated[0] @ qRef

#%% Parameterized Model Order Reduction (PMOR) global approach
# concatenate the modal matrices
# for i, psi in enumerate(psiList):
#     if i == 0:
#         psiConc = psi
#     else:
#         psiConc = np.hstack((psiConc, psi))
# # make a singular value decomposition

#%% Parameterized Model Order Reduction (PMOR) local approach
# def newton_divided_differences(x, y):
#     """
#     Compute Newton divided differences coefficients.
#     x: (n,)
#     y: (n,)
#     """
#     n = len(x)
#     coef = np.array(y, dtype=float)

#     for j in range(1, n):
#         coef[j:n] = (coef[j:n] - coef[j-1:n-1]) / (x[j:n] - x[0:n-j])

#     return coef
# def newton_evaluate(coef, x_data, x):
#     """
#     Evaluate Newton polynomial at x.
#     """
#     n = len(coef)
#     value = coef[n-1]
#     for k in range(n-2, -1, -1):
#         value = value * (x - x_data[k]) + coef[k]
#     return value
# def interpolate_coeffs_newton(snapshots, coeffs, positions):
#     snapshots = np.asarray(snapshots)
#     r_pod, n_modes = coeffs[0].shape
#     n_snap = len(coeffs)

#     assert n_snap == 4, "3rd order Newton needs exactly 4 snapshots"

#     coeffs_interp = []

#     for mu in positions:
#         a_mu = np.zeros((r_pod, n_modes))

#         for i in range(r_pod):
#             for m in range(n_modes):
#                 y = np.array([coeffs[j][i, m] for j in range(n_snap)])
#                 coef = newton_divided_differences(snapshots, y)
#                 a_mu[i, m] = newton_evaluate(coef, snapshots, mu)

#         coeffs_interp.append(a_mu)

#     return coeffs_interp
# def reconstruct_modes(U, coeffs_interp, M=None):
#     Phi_interp = []

#     for a in coeffs_interp:
#         Phi = U @ a

#         if M is not None:
#             for i in range(Phi.shape[1]):
#                 Phi[:, i] /= np.sqrt(Phi[:, i].T @ M @ Phi[:, i])

#         Phi_interp.append(Phi)

#     return Phi_interp
# def project_modes(U, M, Phi_snap):
#     """
#     Project snapshot modes onto POD basis
#     """
#     coeffs = []
#     for Phi in Phi_snap:
#         # a = U^T M Phi
#         a = U.T @ (M @ Phi)
#         coeffs.append(a)
#     return coeffs

# coeffs = project_modes(psiLiftedGlobal, M, psiLiftedSnapshotsList)
# coeffs_interp = interpolate_coeffs_newton(constrainedNodeNumbersSnapshotsList, coeffs, constrainedNodeNumbersConfigurationsList)
# Phi_interp = reconstruct_modes(psiLiftedGlobal, coeffs_interp, M)
# PlotEigenmodes(psiList=Phi_interp,
#                constrainedNodeNumbersList=constrainedNodeNumbersConfigurationsList,
#                nodeNumbersList=nodeNumbersList, nEigenmodesToPlot=nEigenmodesToPlot, figsize=[8, 3])

