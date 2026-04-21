#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#
# Details:  Class(es) that define beam(s) using underlying formulations. For
#           the Euler-Bernulli (EB) beam also reduction (and global reduction!)
#           is supported; special settings for sliding beams and their
#           constraints are supported. This class also natively supports
#           adding the beams (with graphics) to EXUDYN.
# Author:   Sebastian Weyrer
# Date:     2026-04-21
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import time
import numpy as np
import exudyn as exu
from exudyn.utilities import *
import exudyn.graphics as graphics
from scipy.linalg import eigh
import matplotlib.pyplot as plt
from utilities import SetMatplotlibSettings, PrintHeadlineToConsole

# class that defines an euler-bernulli beam, where each node has the DOFs: [axial displacement (x), vertical displacement (y), rotation]
# the class supports computation of eigenmode-snapshots and construction of a global reduction basis
# the class supports modal reduction (reduction basis is initialized as unity matrix)
# "p" are the coordinates in the modal subspace, "q" are the 'real' coordinates
# "unconstrained" always means only for the DOFs that are free
# "constrained" always means only for the DOFs that are constrained
# "lifted" means for all DOFs; the beamTemplate holds the primary information of the beam, while the kwargs can overwrite specific attributes of the beam
class EulerBernulliBeam:
    def __init__(self, beamTemplate=None, **kwargs):
        # retreive the parameters stored in the template
        params = beamTemplate.GetBeamParamatersAsDict().copy()
        params.update(**kwargs) # OVERWRITE partameters if they are specified extra as kwargs
        
        #++++++++++++++++++++++++++++++++++++++++++++++++++
        # set physical parameters of beam
        self.L = params.get('physicsLength')
        self.rhoA = params.get('physicsMassPerLength')
        self.EI = params.get('physicsBendingStiffness')
        self.EA = params.get('physicsAxialStiffness')
        self.m = self.rhoA * self.L
        
        #++++++++++++++++++++++++++++++++++++++++++++++++++
        # set discretization parameters of beam
        self.nElements = params.get('nElements')
        self.lElement = self.L / self.nElements
        self.nNodes = self.nElements + 1
        self.nodeNumbersList = np.arange(0, self.nNodes, 1)
        print('Initializing Euler-Bernulli beam with', self.nNodes, 'nodes.')
        
        #++++++++++++++++++++++++++++++++++++++++++++++++++
        # set reference parameters of beam
        self.nodeReferencePositionsList = np.linspace(0, self.L, self.nNodes)
        self.coordinatesReferenceList = []
        self.positionReference = params.get('positionReference')
        for nodeReferencePosition in self.nodeReferencePositionsList:
            self.coordinatesReferenceList += [nodeReferencePosition + self.positionReference[0], self.positionReference[1], 0]
        self.nDOFsPerNode = 3
        self.nDOFs = self.nNodes * self.nDOFsPerNode
        
        #++++++++++++++++++++++++++++++++++++++++++++++++++
        # set visualization parameters of beam
        self.widthForDrawing = params.get('widthForDrawing')
        self.color = params.get('color')
        
        #++++++++++++++++++++++++++++++++++++++++++++++++++
        # special settings and variable initializations due to sliding joint and force
        self.interpolateBCForSlidingJoint = params.get('interpolateBCForSlidingJoint')
        self.fixBCWhenSliderNotMoving = params.get('fixBCWhenSliderNotMoving')
        self.idxNodeA = 0 # initialize with zero
        SlidingJointFunction = params.get('slidingJointFunction') # this function gives us pos of sliding joint over time
        if SlidingJointFunction is None:
            print('WARNING: Specify a "SlidingJointFunction", unless you only want visualization/initialization!')
            def SlidingJointFunction(t):
                return self.L/2
        self.SlidingJointFunction = SlidingJointFunction
        self.SetConstrainedNodeNumbersDueToSlidingJoint(np.zeros(self.nDOFs), 0) # initialization call
        LoadTipFunction = params.get('loadTipFunction')
        if LoadTipFunction is None:
            print('WARNING: Specify a "LoadTipFunction", unless you only want visualization/initialization!')
            def LoadTipFunction(mbs, t, load):
                return 0
        self.LoadTipFunction = LoadTipFunction
        
        #++++++++++++++++++++++++++++++++++++++++++++++++++
        # assemble the beam
        self.ComputeBeamElementMatrices()
        self.AssembleGlobalMatricesAndAddDamping()
        self.ReduceBeam() # per default, do not reduce
    
    # class function that reduced the beam's matrices
    def ReduceBeam(self, reductionBasis=None):
        if reductionBasis is None:
            self.reductionBasis = np.eye(self.nDOFs)
            self.nDOFsReduced = self.nDOFs
        else:
            self.reductionBasis = reductionBasis
        self.nDOFsReduced = np.shape(self.reductionBasis)[1] # compute nDOFs depending on given reduction basis
        self.MReduced = self.reductionBasis.T @ self.M @ self.reductionBasis
        self.KReduced = self.reductionBasis.T @ self.K @ self.reductionBasis
    
    # class function that diagnoses a reduction basis w.r.t. the resulting reduced Jacobian
    def DiagnoseReductionBasis(self, reductionBasis=None, plot=False, figSize=[5, 5]):
        svList = np.zeros([self.nNodes, 3])
        for node in self.nodeNumbersList:
            # build Jacobian
            C_q = np.zeros([3, self.nDOFs])
            C_q[0, 0] = 1
            C_q[1, 1] = 1
            C_q[2, self.nDOFsPerNode * node + 1] = 1
            # reduce Jacobian
            C_q_Red = C_q @ reductionBasis
            # diagnose reduced Jacobian
            sv = np.linalg.svd(C_q_Red, compute_uv=False)
            svList[node, :] = sv
        if plot:
            SetMatplotlibSettings()
            fig, ax = plt.subplots(1, 1, figsize=figSize)
            for i in range(3):
                ax.semilogy(self.nodeNumbersList, svList[:, i], label=r'$\sigma_' + str(i+1) + '$')
            ax.set_xlabel('node number')
            ax.set_ylabel(r'$\sigma_i(\mathbf{C}_q\boldsymbol{\Psi})$')
            ax.legend()
            ax.grid()
            plt.tight_layout()
        return svList
        
    # class function that returns the mass and stiffness matrices for one Bernoulli-Euler beam element
    def ComputeBeamElementMatrices(self):
        # reference used: https://people.duke.edu/~hpgavin/StructuralDynamics/StructuralElements.pdf
        k = np.zeros([self.nDOFsPerNode*2, self.nDOFsPerNode*2])
        m = np.zeros([self.nDOFsPerNode*2, self.nDOFsPerNode*2])
        
        #++++++++++++++++++++++++++++++++++++++++++++++++++
        # stiffness matrix
        # axial stiffness
        k[0,0] = k[3,3] = self.EA / self.lElement
        k[0,3] = k[3,0] = -self.EA / self.lElement
        # bending stiffness
        k[1,1] = k[4,4] = 12 * self.EI / self.lElement**3
        k[1,4] = k[4,1] = -12 * self.EI / self.lElement**3
        k[1,2] = k[2,1] = 6 * self.EI / self.lElement**2
        k[1,5] = k[5,1] = 6 * self.EI / self.lElement**2
        k[4,2] = k[2,4] = -6 * self.EI / self.lElement**2
        k[4,5] = k[5,4] = -6 * self.EI / self.lElement**2
        k[2,2] = k[5,5] = 4 * self.EI / self.lElement
        k[2,5] = k[5,2] = 2 * self.EI / self.lElement
        
        #++++++++++++++++++++++++++++++++++++++++++++++++++
        # mass matrix
        m = (self.rhoA*self.lElement) / 420 * np.array([[140, 0, 0, 70, 0, 0],
                                                        [0, 156, 22*self.lElement, 0, 54, -13*self.lElement],
                                                        [0, 22*self.lElement, 4*self.lElement**2, 0, 13*self.lElement, -3*self.lElement**2],
                                                        [70, 0, 0, 140, 0, 0],
                                                        [0, 54, 13*self.lElement, 0, 156, -22*self.lElement],
                                                        [0, -13*self.lElement, -3*self.lElement**2, 0, -22*self.lElement, 4*self.lElement**2]])            
        self.m = m
        self.k = k
    
    # class function that assembles the global mass and stiffness matrix of the Bernulli-Euler beam
    def AssembleGlobalMatricesAndAddDamping(self):
        M = np.zeros([self.nDOFs, self.nDOFs])
        K = np.zeros([self.nDOFs, self.nDOFs])
        for i in range(self.nElements):
            # always three DOFs are overlapping
            indexList = [self.nDOFsPerNode*i,
                         self.nDOFsPerNode*i+1,
                         self.nDOFsPerNode*i+2,
                         self.nDOFsPerNode*i+3,
                         self.nDOFsPerNode*i+4,
                         self.nDOFsPerNode*i+5]
            # make superposition
            for j in range(self.nDOFsPerNode*2): # two nodes per element
                for k in range(self.nDOFsPerNode*2): # two nodes per element
                    M[indexList[j], indexList[k]] += self.m[j, k]
                    K[indexList[j], indexList[k]] += self.k[j, k]
        self.M = M
        self.K = K
    
    # class function that enriches a reduction basis
    def EnrichReductionBasis(self, reductionBasis=None):
        # for every (possibly) constrained DOF, add unity vector with 1 for the constrained DOF
        for node in self.nodeNumbersList:
            e = np.zeros([self.nDOFs, 1])
            if node == 0:
                e[0, 0] = 1
                E = e
                e = np.zeros([self.nDOFs, 1])
                e[1, 0] = 1
            else:
                e[node*self.nDOFsPerNode+1, 0] = 1
            E = np.hstack((E, e))
        reductionBasisConc = np.hstack((reductionBasis, E))
        U, S, Vt = np.linalg.svd(reductionBasisConc, full_matrices=False)
        rank = np.sum(S > 1e-10 * S[0])
        reductionBasisEnriched = U[:, :rank]
        print('Enriched reduction basis leads to', rank, 'modal coordinates.')
        return reductionBasisEnriched
        
    # class function to internally set the node number that is currently constrained depending on the sliding joint function
    # note that this function receives the non reduced coordinates of the beam
    # note that this functions assumes that the sliding joint is between node A and node B; alpha weights between A and B
    # def SetConstrainedNodeNumbersDueToSlidingJoint(self, q, t):
    #     positionListToSetConstrained = self.nodeReferencePositionsList
    #     if False: # this can cause instabilities!
    #         qTotal = q + np.array(self.coordinatesReferenceList)
    #         qXTotal = qTotal[::3]
    #         positionListToSetConstrained = qxTotal
        
    #     #++++++++++++++++++++++++++++++++++++++++++++++++++
    #     # determine whether joint is moving or not
    #     if self.fixBCWhenSliderNotMoving:
    #         jointMoving = False # per default, the joint is not moving
    #         if abs(self.SlidingJointFunction(t+1e-6)-self.SlidingJointFunction(t)) >= 1e-12:
    #             jointMoving = True
    #     else:
    #         jointMoving = True # when not fixed, joint is always moving (per default)
        
    #     #++++++++++++++++++++++++++++++++++++++++++++++++++
    #     # compute node A and B and suggestion for alpha
    #     idxNodeA = np.where(positionListToSetConstrained < self.SlidingJointFunction(t))[0][-1] # LAST element that is smaller than current pos
    #     # to avoid node flickering, check whether idxNodeA is increasing, otherwise, do not change
    #     # if joint not moving, idxNodeA only can increase for the interpolation case
    #     if not self.interpolateBCForSlidingJoint or not jointMoving:
    #         if idxNodeA > self.idxNodeA:
    #             pass
    #         else:
    #             idxNodeA = self.idxNodeA # overwrite, such that no jumping back is possible
    #     if idxNodeA > (self.nNodes-2):
    #         idxNodeA = self.nNodes - 2 # such that we do not slide beyond the beam
    #     self.idxNodeB = idxNodeA + 1
    #     self.idxNodeA = idxNodeA
    #     posNodeA = positionListToSetConstrained[self.idxNodeA]
    #     posNodeB = positionListToSetConstrained[self.idxNodeB]
        
    #     #++++++++++++++++++++++++++++++++++++++++++++++++++
    #     # compute recommendation for alpha that is used depending on BC interpolation method
    #     alpha = np.clip((self.SlidingJointFunction(t)-posNodeA)/(posNodeB-posNodeA), 0, 1) # bounded
        
    #     #++++++++++++++++++++++++++++++++++++++++++++++++++
    #     # pin to one node if either joint is not moving OR we do NOT interpolate
    #     if not self.interpolateBCForSlidingJoint or not jointMoving:
    #         if alpha >= 0.5:
    #             alpha = 1
    #         else:
    #             alpha = 0
    #     # and compute hermite weight
    #     self.w = 3*alpha**2 - 2*alpha**3
    
    def SetConstrainedNodeNumbersDueToSlidingJoint(self, q, t):
        positionListToSetConstrained = self.nodeReferencePositionsList
        # compute node A and B and suggestion for alpha
        idxNodeA = np.where(positionListToSetConstrained < self.SlidingJointFunction(t))[0][-1] # LAST element that is smaller than current pos
        self.idxNodeB = idxNodeA + 1
        self.idxNodeA = idxNodeA
        posNodeA = positionListToSetConstrained[self.idxNodeA]
        posNodeB = positionListToSetConstrained[self.idxNodeB]
        
        #++++++++++++++++++++++++++++++++++++++++++++++++++
        # compute recommendation for alpha that is used depending on BC interpolation method
        self.alpha = np.clip((self.SlidingJointFunction(t)-posNodeA)/(posNodeB-posNodeA), 0, 1) # bounded
        
        
    # class function that returns the (residuums) of the constraint euqtions, either in 'normal' or reduced coordinates
    # the connector holds q for both markers (i.e. objects) it connects
    # since the ground marker does not hold coordinates, here it only holds the coordinates of the ODE2 object
    # note that this function actually receives the modally reduced coordinates p
    # def UFConstrained(self, mbs, t, itemNumber, p, p_t, velcoityLevel):
    #     q = self.reductionBasis @ p # map the modal coordinates to 'real' ones
    #     self.SetConstrainedNodeNumbersDueToSlidingJoint(q, t)
    #     g = [] # initialize the constraint vector
    #     g += [q[0]-0] # first node, x coordinate is constrained
    #     g += [q[1]-0] # first node, y coordinate is constrained
    #     g += [(1-self.w)*q[self.nDOFsPerNode*self.idxNodeA+1] + self.w*q[self.nDOFsPerNode*self.idxNodeB+1]]
    #     return g
    
    def UFConstrained(self, mbs, t, itemNumber, p, p_t, velcoityLevel):
        q = self.reductionBasis @ p # map the modal coordinates to 'real' ones
        self.SetConstrainedNodeNumbersDueToSlidingJoint(q, t)
        # Hermite shape functions
        N2 = 1 - 3*self.alpha**2 + 2*self.alpha**3        # y at node A
        N3 = (self.alpha - 2*self.alpha**2 + self.alpha**3) * self.lElement  # rotation at node A
        N5 = 3*self.alpha**2 - 2*self.alpha**3            # y at node B
        N6 = (-self.alpha**2 + self.alpha**3) * self.lElement        # rotation at node B
        idxA = self.nDOFsPerNode * self.idxNodeA
        idxB = self.nDOFsPerNode * self.idxNodeB
        g = [] # initialize the constraint vector
        g += [q[0]-0] # first node, x coordinate is constrained
        g += [q[1]-0] # first node, y coordinate is constrained
        g += [N2*q[idxA+1] + N3*q[idxA+2] + N5*q[idxB+1] + N6*q[idxB+2]]
        return g
    
    # class function that returns the Jacobian of the constraints, either in 'normal' or reduced coordinates
    # the connector holds q for both markers (i.e. objects) it connects
    # since the ground marker does not hold coordinates, here it only holds the coordinates of the ODE2 object
    # non transposed joint equations C derived by q: C_q (nAE x 2q)
    # the transposed C_qT is multiplied from the left to the Lagrangians --> gives us a 2q vector
    # note that this function actually receives the modally reduced coordinates p
    # def UFJacobian(self, mbs, t, itemNumber, p, p_t, velocityLevel):
    #     # here we first of all set up the Jacobian in full coordinates
    #     C_q = np.zeros([3, self.nDOFs])
    #     C_q[0, 0] = 1 # first AE is g[0] = q[0] - 0
    #     C_q[1, 1] = 1 # second AE is g[1] = q[1] - 0
    #     C_q[2, self.nDOFsPerNode*self.idxNodeA+1] = (1-self.w)
    #     C_q[2, self.nDOFsPerNode*self.idxNodeB+1] = self.w
    #     # now we reduce the Jacobian
    #     C_q_Red = C_q @ self.reductionBasis
    #     return C_q_Red
    
    def UFJacobian(self, mbs, t, itemNumber, p, p_t, velocityLevel):
        N2 = 1 - 3*self.alpha**2 + 2*self.alpha**3
        N3 = (self.alpha - 2*self.alpha**2 + self.alpha**3) * self.lElement
        N5 = 3*self.alpha**2 - 2*self.alpha**3
        N6 = (-self.alpha**2 + self.alpha**3) * self.lElement
        
        idxA = self.nDOFsPerNode * self.idxNodeA
        idxB = self.nDOFsPerNode * self.idxNodeB
    
    
        # here we first of all set up the Jacobian in full coordinates
        C_q = np.zeros([3, self.nDOFs])
        C_q[0, 0] = 1 # first AE is g[0] = q[0] - 0
        C_q[1, 1] = 1 # second AE is g[1] = q[1] - 0
        C_q[2, idxA+1] = N2
        C_q[2, idxA+2] = N3
        C_q[2, idxB+1] = N5
        C_q[2, idxB+2] = N6
        # now we reduce the Jacobian
        C_q_Red = C_q @ self.reductionBasis
        return C_q_Red
    
    # class function that adds a force on the tip of the beam
    # note that this function actually receives the modally reduced coordinates p
    def UFForce(self, mbs, t, itemNumber, p, p_t):
        f = np.zeros(self.nDOFs)
        f[-2] = self.LoadTipFunction(None, t, None)
        fRed = self.reductionBasis.T @ f
        return fRed
    
    # class function that adds the beam to EXUDYN's mbs such that it can be used therein
    # this function either adds the reduced beam or the non-reduced - everything automatcially
    # if no reduction is applied, the full quantities are hold within the reduced ones
    def AddBeamToMBSAndAddSensors(self, mbs, mCoordsGround=None, show=False):
        nGenericBeam = mbs.AddNode(NodeGenericODE2(referenceCoordinates=[0]*self.nDOFsReduced,
                                                   initialCoordinates=[0]*self.nDOFsReduced,
                                                   initialCoordinates_t=[0]*self.nDOFsReduced,
                                                   numberOfODE2Coordinates=self.nDOFsReduced))
        sCoordinates = mbs.AddSensor(SensorNode(nodeNumber=nGenericBeam, outputVariableType=exu.OutputVariableType.Coordinates, storeInternal=True))
        sCoordinates_t = mbs.AddSensor(SensorNode(nodeNumber=nGenericBeam, outputVariableType=exu.OutputVariableType.Coordinates_t, storeInternal=True))
        # add super element object
        oODE2Beam = mbs.AddObject(ObjectGenericODE2(nodeNumbers=[nGenericBeam],
                                                    massMatrix=self.MReduced,
                                                    stiffnessMatrix=self.KReduced,
                                                    forceUserFunction=self.UFForce,
                                                    visualization=VObjectGenericODE2(graphicsDataUserFunction=self.BeamGraphics, show=show)))
        mCoordsBeam = mbs.AddMarker(MarkerObjectODE2Coordinates(objectNumber=oODE2Beam))
        ccBeam = mbs.AddObject(CoordinateVectorConstraint(markerNumbers=[mCoordsGround, mCoordsBeam],
                                                          scalingMarker0=np.zeros([3, 1]), # always three AE (first node x and y, other constrained node y)
                                                          constraintUserFunction=self.UFConstrained,
                                                          jacobianUserFunction=self.UFJacobian,
                                                          velocityLevel=False))
        return sCoordinates, sCoordinates_t
        
    # class function to get GLOBAL mass normalized list of eigenmodes
    def GetGlobalMassNormalizedLiftedEigenmodesList(self, eigenmodesList=[]):
        nEigenmodesSets = len(eigenmodesList)
        eigenmodesMassNormalizedList = []
        # do GLOBAL mass normalization
        for i, psi in enumerate(eigenmodesList):
            nEigenmodes = np.shape(psi)[1]
            psiMassNormalized = np.zeros_like(psi)
            for j in range(nEigenmodes):
                normFactor = np.sqrt(psi[:, j].T @ self.M @ psi[:, j])
                psiMassNormalized[:, j] = psi[:, j] / normFactor
            eigenmodesMassNormalizedList += [psiMassNormalized]
        return eigenmodesMassNormalizedList
        
    # class function to compute eigenmodes for given number of y-direction constrained nodes
    # the DOF in x and y direction of the first node is always automatically constrained
    # you can additionally also constrain the x DOF of the node
    # if one node is constrained, the configuration analysed is with the middle node being constrained
    # if two nodes are constrained, the configurations analysed is with second and last node constrained
    # if more than two nodes are constrained, use equally spaced configurations between second and last node
    def GetGlobalMassNormalizedEigenmodesListForNumberOfConstrainedNodes(self, nNodesConstrained=2, nEigenmodes=1,
                                                                         constrainXDOFOfConstrainedNodes=False,
                                                                         nodeNumbersConstrainedList=None, verbose=False):
        if verbose:
            PrintHeadlineToConsole(text='Computing snapshots')
        self.eigenmodesUnconstrainedList = [] # list of eigenmodes (without constrained BCs) PER constrained node number
        self.eigenmodesLiftedList = [] # list of eigenmodes (with constrained DOFs) PER constrained node number; not post-processed
        self.MUnconstrainedList = [] # list of mass matrices respecting the BCs PER constrained node number
        self.KUnconstrainedList = [] # stiffness matrices respecting the BCs PER constrained node number
        self.DOFsUnconstrainedList = [] # list of free DOFs PER constrained node number
        self.DOFsConstrainedList = [] # list of constrained DOFs per constrained node number
        if nodeNumbersConstrainedList == None:
            self.nNodesConstrained = nNodesConstrained
            if self.nNodesConstrained <= 0:
                raise ValueError('"nNodesConstrained" must be greater than 0.')
            elif self.nNodesConstrained == 1:
                self.nodeNumbersConstrainedList = [self.nodeNumbersList[self.nNodes//2]]
            elif self.nNodesConstrained == 2:
                self.nodeNumbersConstrainedList = [self.nodeNumbersList[1]]
                self.nodeNumbersConstrainedList += [self.nodeNumbersList[-1]]
            else:
                indicesOfNodesForSnapshots = np.linspace(1, self.nNodes-1, self.nNodesConstrained, dtype=int)
                self.nodeNumbersConstrainedList = self.nodeNumbersList[indicesOfNodesForSnapshots]
        else:
            print('WARNING: Specific nodes to be constrained are given - ignoring "nNodesConstrained".')
            self.nNodesConstrained = len(nodeNumbersConstrainedList)
            self.nodeNumbersConstrainedList = nodeNumbersConstrainedList
        if verbose:
            print('Computing eigenmodes for constrained node numbers (in addition to first node):')
        for i, nodeNumberConstrained in enumerate(self.nodeNumbersConstrainedList):
            if verbose:
                print('Snapshot ' + str(i+1) + ':', nodeNumberConstrained)
        startTime = time.time()
        for i, nodeNumberConstrained in enumerate(self.nodeNumbersConstrainedList):
            # now process every set
            DOFsConstrained = [0, 1] # movement of first node is always constrained
            if constrainXDOFOfConstrainedNodes:
                DOFsConstrained += [nodeNumberConstrained * self.nDOFsPerNode] # x direction is constrained
            DOFsConstrained += [nodeNumberConstrained * self.nDOFsPerNode + 1] # y direction is constrained
            DOFsUnconstrained = np.setdiff1d(np.arange(self.nDOFs), DOFsConstrained) # get list of free DOFs
            self.DOFsConstrainedList += [DOFsConstrained]
            self.DOFsUnconstrainedList += [DOFsUnconstrained]
            KUnconstrained = self.K[np.ix_(DOFsUnconstrained, DOFsUnconstrained)]
            MUnconstrained = self.M[np.ix_(DOFsUnconstrained, DOFsUnconstrained)]
            self.KUnconstrainedList += [KUnconstrained]
            self.MUnconstrainedList += [MUnconstrained]
            # solve the eigenvalues problem
            eigvals, eigvecs = eigh(KUnconstrained, MUnconstrained, subset_by_index=[0, nEigenmodes-1])
            if verbose:
                print('Eigenvalues ' + str(i+1) + ':', eigvals)
            self.eigenmodesUnconstrainedList += [eigvecs]
            # now fill the constrained DOFs with 0
            nDOFsSkipped = 0
            psi = np.zeros([self.nDOFs, nEigenmodes])
            for DOF in range(self.nDOFs):
                if DOF in DOFsUnconstrained:
                    if eigvecs.ndim == 1:
                        psi[DOF, :] = eigvecs[DOF-nDOFsSkipped]
                    elif eigvecs.ndim == 2:
                        psi[DOF, :] = eigvecs[DOF-nDOFsSkipped, :]
                if DOF in DOFsConstrained:
                    nDOFsSkipped += 1
            self.eigenmodesLiftedList += [psi]
        computationTime = time.time() - startTime
        if verbose:
            print('Successfully computed', len(self.eigenmodesLiftedList), 'set(s) of', nEigenmodes, 'eigenmode(s).')
            print('The modal matrix/matrices in the list has/have size:', np.shape(self.eigenmodesLiftedList[0]))
            print('Computation time: ' + str(computationTime) + 's.')
            print('(GLOBAL) mass normalize the computed set(s) of eigenmode(s).')
        return self.GetGlobalMassNormalizedLiftedEigenmodesList(eigenmodesList=self.eigenmodesLiftedList)
    
    # class function to plot eigenmodes list, corresponding to constrained nodes list
    # the first node is always automatically drawn constrained
    def PlotEigenmodesSets(self, eigenmodesList=[], nodeNumbersConstrainedList=[],
                           nEigenmodesToPlot=3, figSize=[5, 5], easyPlotting=True):
        SetMatplotlibSettings()
        nEigenmodesSets = len(eigenmodesList)
        offset = 1
        if not easyPlotting:
            offset = 0
        fig, axs = plt.subplots(nEigenmodesSets, self.nDOFsPerNode-offset, sharex=True, figsize=figSize)
        if nEigenmodesSets == 1:
            axs = np.array([axs])
        linesForLegend = []
        labelsForLegend = []
        for i in range(nEigenmodesSets):
            for j in range(nEigenmodesToPlot):
                if not easyPlotting:
                    line, = axs[i, 0].plot(self.nodeNumbersList, eigenmodesList[i][::self.nDOFsPerNode, j])
                    markerStr = '-'
                else:
                    if j == 0:
                        markerStr = '*-'
                    elif j == 1:
                        markerStr = 'o-'
                    elif j == 2:
                        markerStr = 'x-'
                line, = axs[i, 1-offset].plot(self.nodeNumbersList, eigenmodesList[i][1::self.nDOFsPerNode, j], markerStr, markevery=10)
                line, = axs[i, 2-offset].plot(self.nodeNumbersList, eigenmodesList[i][2::self.nDOFsPerNode, j], markerStr, markevery=10)
                if i == 0:
                    linesForLegend.append(line)
                    labelsForLegend.append('mode ' + str(j+1))
            if not easyPlotting:
                axs[i, 0].plot(0, 0, 'o', color='k')
                axs[i, 0].plot(nodeNumbersConstrainedList[i], 0, 'o', color='k')
            axs[i, 1-offset].plot(0, 0, 'o', color='k')
            axs[i, 1-offset].plot(nodeNumbersConstrainedList[i], 0, 'o', color='k')
            axs[i, 2-offset].plot(0, 0, 'o', color='k')
            axs[i, 2-offset].plot(nodeNumbersConstrainedList[i], 0, 'o', color='k')
            
            if not easyPlotting:
                axs[i, 0].grid()
            axs[i, 1-offset].grid()
            axs[i, 2-offset].grid()
            if not easyPlotting:
                axs[i, 0].set_ylabel('x def. component')
            if easyPlotting and i == 1:
                axs[i, 1-offset].set_ylabel('y def. component')
                axs[i, 2-offset].set_ylabel('rot. component')
        fig.legend(linesForLegend, labelsForLegend, loc='upper center',
                   ncol=nEigenmodesToPlot, bbox_to_anchor=(0.5, 0.97))
        if not easyPlotting:
            axs[-1, 0].set_xlabel('node number')
        axs[-1, 1-offset].set_xlabel('node number')
        axs[-1, 2-offset].set_xlabel('node number')
        plt.subplots_adjust(right=0.8, left=0.2, top=0.9, wspace=0.6)
        plt.tight_layout(rect=[0, 0, 1, 0.9])
        return fig
    
    # class function to get global reduction basis of beam
    def GetGlobalReductionBasisUsingEigenmodesList(self, eigenmodesList=[], nGlobalModes=24, verbose=False):
        if verbose:
            PrintHeadlineToConsole(text='Computing global reduction basis')
        for i, psi in enumerate(eigenmodesList):
            if i == 0:
                psiConc = psi
            else:
                psiConc = np.hstack((psiConc, psi))
        maxGlobalModes = np.shape(psiConc)[1]
        if verbose:
            print('The global reduction basis could have', maxGlobalModes, 'mode(s).')
        if nGlobalModes > maxGlobalModes:
            raise ValueError('The specified number of global modes exceeds the max. possible number of ' + str(maxGlobalModes) + '.')
        # perform SVD
        startTime = time.time()
        U, S, Vt = np.linalg.svd(psiConc, full_matrices=False)
        psiGlobal = U[:, :nGlobalModes] # select number of global modes
        computationTime = time.time() - startTime
        if verbose:
            print('Computed global reduction basis with ' + str(np.shape(psiGlobal)[1]) + ' global mode(s).')
            print('Computation time: ' + str(computationTime) + 's.')
        return psiGlobal
    
    # class function that computes the energy (kinetic and strain and potential) of the system over time (given time series of minimal coordinates)
    def GetKineticAndStrainAndPotentialBeamEnergy(self, q, q_t, sensorWritePeriod=0):
        nTimeSteps = np.shape(q)[0]
        E = np.zeros(nTimeSteps)
        for i in range(nTimeSteps):
            # we compute energy always in NON-REDUCED coordinates q
            E[i] = (1/2) * (q_t[i, :] @ self.M @ q_t[i, :].T) # kinetic energy
            E[i] += (1/2) * (q[i, :] @ self.K @ q[i, :].T) # strain energy
            E[i] -= self.LoadTipFunction(None, i*sensorWritePeriod, None) * q[i, -self.nDOFsPerNode+1] # load potential
        return E
    
    # class function that returns a graphics data list depending on the state of the mbs; to be used within EXUDYN
    # this function is used in the ODE2 object
    def BeamGraphics(self, mbs, itemNumber):
        # use super element function
        p = mbs.GetObjectOutputSuperElement(objectNumber=itemNumber, variableType=exu.OutputVariableType.Coordinates,
                                            meshNodeNumber=0, configuration=exu.ConfigurationType.Current)
        q = self.reductionBasis @ p # map back to 'real' space
        qTotal = q + np.array(self.coordinatesReferenceList)
        graphicsData = []
        for i in range(self.nNodes-1):
            x0 = qTotal[i*self.nDOFsPerNode]
            x1 = qTotal[(i+1)*self.nDOFsPerNode]
            y0 = qTotal[i*self.nDOFsPerNode + 1]
            y1 = qTotal[(i+1)*self.nDOFsPerNode + 1]
            phi0 = qTotal[i*self.nDOFsPerNode + 2] # here, anyways, no reference is added
            phi1 = qTotal[(i+1)*self.nDOFsPerNode + 2] # here, anyways, no reference is added
            x0Normal = RotationMatrixZ(phi0) @ np.array([0, self.widthForDrawing/2, 0])
            x1Normal = RotationMatrixZ(phi1) @ np.array([0, self.widthForDrawing/2, 0])
            p00 = np.array([x0, y0, self.positionReference[2]]) - x0Normal
            p10 = np.array([x1, y1, self.positionReference[2]]) - x1Normal
            p11 = np.array([x1, y1, self.positionReference[2]]) + x1Normal
            p01 = np.array([x0, y0, self.positionReference[2]]) + x0Normal
            graphicsData += [graphics.Quad(pList=[p00, p10, p11, p01], color=self.color)]
        return graphicsData
    
    # class function that globally reduces the beam (it is a wrapper function that calls underlying methods of this beam class)
    def GlobalReduceBeamWrapper(self, nGlobalModes=24, nSnapshots=6, nEigenmodesPerSnapshot=6, plot=False, figSize=[5, 7], verbose=False):
        eigenmodesLiftedAndMassNormalizedListPMR = self.GetGlobalMassNormalizedEigenmodesListForNumberOfConstrainedNodes(nNodesConstrained=nSnapshots,
                                                                                                                         nEigenmodes=nEigenmodesPerSnapshot,
                                                                                                                         verbose=verbose)
        if plot:
            self.PlotEigenmodesSets(eigenmodesList=eigenmodesLiftedAndMassNormalizedListPMR,
                                       nEigenmodesToPlot=min(nEigenmodesPerSnapshot, 3),
                                       nodeNumbersConstrainedList=self.nodeNumbersConstrainedList, figSize=figSize)
        globalReductionBasis = self.GetGlobalReductionBasisUsingEigenmodesList(eigenmodesList=eigenmodesLiftedAndMassNormalizedListPMR,
                                                                                 nGlobalModes=nGlobalModes, verbose=verbose)
        if plot:
            fig, ax = plt.subplots()
            im = ax.imshow(globalReductionBasis, cmap='viridis', aspect='auto')
            fig.colorbar(im, ax=ax)
            ax.set_xlabel("Columns")
            ax.set_ylabel("Rows")
            plt.show()
        # self.DiagnoseReductionBasis(reductionBasis=globalReductionBasis, plot=plot)
        # the bug was that the constrained node number flickered, however, also enrichment of reduction basis is now supported
        # globalReductionBasisEnriched = beamPMR.EnrichReductionBasis(reductionBasis=globalReductionBasis)
        # beamPMR.DiagnoseReductionBasis(reductionBasis=globalReductionBasisEnriched, plot=True)
        self.ReduceBeam(reductionBasis=globalReductionBasis)
    
    # class function to get info about 'time' and the coordinates q and q_t with given sensor numbers
    def GetTimeAndCoordinates(self, mbs, sCoordinates, sCoordinates_t):
        time = mbs.GetSensorStoredData(sensorNumber=sCoordinates)[:, 0]
        nTimeSteps = np.size(time)
        q = np.zeros([nTimeSteps, self.nDOFs])
        q_t = np.zeros([nTimeSteps, self.nDOFs])
        for i in range(nTimeSteps): # for each timestep, project back
            q[i, :] = self.reductionBasis @ mbs.GetSensorStoredData(sensorNumber=sCoordinates)[i, 1:]
            q_t[i, :] = self.reductionBasis @ mbs.GetSensorStoredData(sensorNumber=sCoordinates_t)[i, 1:]
        return nTimeSteps, time, q, q_t

# class that is used as a template for setting up a beam
# this class just holds the parameters that are needed for setting up the beam
# specific parameters can easily be overwritten but this template holds ALL parameters that can be used to set up a beam
class BeamTemplate:
    def __init__(self, physicsLength=1, physicsMassPerLength=5, physicsBendingStiffness=3e4,
                 physicsAxialStiffness=6e6, nElements=10, widthForDrawing=0.2, color=graphics.color.blue,
                 positionReference=[0, 0, 0], slidingJointFunction=None, loadTipFunction=None, interpolateBCForSlidingJoint=True,
                 fixBCWhenSliderNotMoving=False):
        self.physicsLength = physicsLength
        self.physicsMassPerLength = physicsMassPerLength
        self.physicsBendingStiffness = physicsBendingStiffness
        self.physicsAxialStiffness = physicsAxialStiffness
        self.nElements = nElements
        self.widthForDrawing = widthForDrawing
        self.color = color
        self.positionReference = positionReference
        self.slidingJointFunction = slidingJointFunction
        self.loadTipFunction = loadTipFunction
        self.interpolateBCForSlidingJoint = interpolateBCForSlidingJoint
        self.fixBCWhenSliderNotMoving = fixBCWhenSliderNotMoving
    
    def GetBeamParamatersAsDict(self):
        return vars(self)