import numpy as np
import os
import sys
import re
import time
from readInp import getFort, parsInp

# Read input
input_file = sys.argv[1]
molecule, diis_mode = parsInp(input_file)

#########################################################################
######################   Working CCSD Equations    ######################
#########################################################################

# Get number of orbitals, SCF energy, Fock matrix, MO coefficients
O, V, NB, scfE, Fock, Coeff = getFort(molecule)

#########################################################################
##########################   Intermediate Terms    ######################
#########################################################################

class Intermediate:
  
  def tau_tildeEq(T, O, V, t1, t2):
    # tau_tilde intermediate for CCSD T equations
    if T==1:
      tau_tilde = t2 + 0.5*(np.einsum('ia,jb->ijab',t1,t1,optimize=True) - np.einsum('ib,ja->ijab',t1,t1,optimize=True))
    return tau_tilde
  
  def tauEq(T, O, V, t1, t2):
    # tau intermediate for CCSD T equations
    if T==1:
      tau = t2 + np.einsum('ia,jb->ijab',t1,t1,optimize=True) - np.einsum('ib,ja->ijab',t1,t1,optimize=True)
    return tau
  
  def intermediateEqs(T, O, V, Fock, t1, t2, IJKL, ABCD, IABC, IJAB, IAJB, IJKA, tau_tilde, tau):
    # F and W intermediates for CCSD T equations
    O2=2*O
    V2=2*V
    if T==1:
      # F_ae
      F_ae = np.zeros((V2, V2))
      F_ae += (1 - np.eye(V2)) * Fock[O2:, O2:]
      F_ae = -0.5 * np.einsum('me,ma->ae', Fock[:O2, O2:], t1, optimize=True)
      F_ae += np.einsum('mf,mafe->ae', t1, IABC, optimize=True)
      F_ae -= 0.5 * np.einsum('mnaf,mnef->ae', tau_tilde, IJAB, optimize=True)    
      # F_mi
      F_mi=np.zeros((O2, O2))
      F_mi += (1 - np.eye(O2)) * Fock[:O2, :O2]
      F_mi += 0.5 * np.einsum('ie,me->mi', t1, Fock[:O2, O2:], optimize=True)
      F_mi += np.einsum('ne,mnie->mi', t1, IJKA, optimize=True)
      F_mi += 0.5 * np.einsum('inef,mnef->mi', tau_tilde, IJAB, optimize=True)
      # F_me
      F_me = np.zeros((O2, V2))
      F_me = np.copy(Fock[:O2, O2:])
      F_me += np.einsum('nf,mnef->me', t1, IJAB, optimize=True)
      # W_mnij
      W_mnij = np.copy(IJKL)
      W_mnij += np.einsum('je,mnie->mnij', t1, IJKA, optimize=True)
      W_mnij -= np.einsum('ie,mnje->mnij', t1, IJKA, optimize=True)
      W_mnij += 0.5 * np.einsum('ijef,mnef->mnij', tau, IJAB, optimize=True)
      # W_abef
      W_abef =np.copy(ABCD)
      W_abef += np.einsum('mb,maef->abef',t1,IABC,optimize=True)
      W_abef -= np.einsum('ma,mbef->abef',t1,IABC,optimize=True)
      # W_mbej
      W_mbej = np.copy(-np.transpose(IAJB, axes=(0,1,3,2)))
      W_mbej += np.einsum('jf,mbef->mbej', t1, IABC, optimize=True)
      W_mbej += np.einsum('nb,mnje->mbej', t1, IJKA, optimize=True)
      W_mbej -= 0.5 * np.einsum('jnfb,mnef->mbej', t2, IJAB, optimize=True)
      W_mbej -= np.einsum('jf,nb,mnef->mbej', t1, t1, IJAB, optimize=True)
    return F_ae, F_mi, F_me, W_mnij, W_abef, W_mbej

#########################################################################
##########################   Amplitude Equations    #####################
#########################################################################

class Amplitudes:

  def t1Eq(T, O, Fock, t1, t2, IABC, IJKA, IAJB, F_ae, F_mi, F_me, D1):
    O2=2*O
    V2=2*V
    # CCSD T1 amplitude equation
    if T==1:
      t1_f = np.copy(Fock[:O2, O2:])  
      t1_f += np.einsum('ie,ae->ia', t1, F_ae, optimize=True)
      t1_f -= np.einsum('ma,mi->ia', t1, F_mi, optimize=True)
      t1_f += np.einsum('imae,me->ia', t2, F_me, optimize=True)
      t1_f -= 0.5 * np.einsum('imef,maef->ia', t2, IABC,optimize=True)
      t1_f += 0.5 * np.einsum('mnae,nmie->ia', t2, IJKA,optimize=True)
      t1_f -= np.einsum('nf,naif->ia', t1, IAJB,optimize=True)
    # Divide by energy denominator
      t1_f /= D1
    return t1_f
  
  ###########################################################################
  def t2Eq(T, t1, t2, IABC, IJAB, IJKA, IAJB, tau, F_ae, F_mi, F_me, W_mnij, W_abef, W_mbej, D2):
    if T==1:
    # CCSD T2 amplitude equation
      # P(ab) terms
      X1 = F_ae - 0.5*np.einsum('mb,me->be',t1,F_me,optimize=True)
      X2 = np.einsum('ijae,be->ijab',t2,X1,optimize=True) - np.einsum('ma,ijmb->ijab',t1,IJKA,optimize=True)
      t2_f = IJAB + X2 - np.transpose(X2,axes=(0,1,3,2))
      del X1, X2
      # P(ij) terms
      X1 = F_mi + 0.5*np.einsum('je,me->mj',t1,F_me,optimize=True)
      X2 = - np.einsum('imab,mj->ijab',t2,X1,optimize=True) - np.einsum('ie,jeab->ijab',t1,IABC,optimize=True)
      t2_f += X2 - np.transpose(X2,axes=(1,0,2,3))
      del X1, X2
      # P(ij,ab) terms
      X1 = np.einsum('ie,mbje->mbij',t1,IAJB,optimize=True)
      X2 = np.einsum('imae,mbej->ijab',t2,W_mbej,optimize=True) + np.einsum('ma,mbij->ijab',t1,X1,optimize=True)
      t2_f += X2 - np.transpose(X2,axes=(1,0,2,3)) - np.transpose(X2,axes=(0,1,3,2))  + np.transpose(X2,axes=(1,0,3,2))
      del X1, X2
      # tau terms
      t2_f += 0.5*np.einsum('ijef,abef->ijab',tau,W_abef,optimize=True) + 0.5*np.einsum('mnab,mnij->ijab',tau,W_mnij,optimize=True)
      # Divide by energy denominator
      t2_f /= D2    
    return t2_f

#########################################################################
############################   Energy Equation    #######################
#########################################################################
class Energy:

  def E_CCSD(O, Fock, t1, IJAB, tau):
    # CCSD energy
    O2 = 2*O
    E_Corr2_1 = np.einsum('ia,ia->', t1, Fock[:O2, O2:])
    E_Corr2_2 = 0.25 * np.einsum('ijab,ijab->', tau, IJAB)
    E_Corr2 = E_Corr2_1 + E_Corr2_2
    return E_Corr2
