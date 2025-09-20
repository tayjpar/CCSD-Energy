import numpy as np
import os
import sys
import re
import time
from readInp import parsInp, getFort, get2e, conMO
from ccsdEqs import Intermediate, Amplitudes, Energy, molecule
from diis import Extrapolation

# Read input
input_file = sys.argv[1]
molecule, diis_mode = parsInp(input_file)

# Get number of orbitals, SCF energy, Fock matrix, MO coefficients
O, V, NB, scfE, Fock, Coeff = getFort(molecule)

# Initialize arrays
coul = np.zeros((NB,NB,NB,NB))
exc = np.zeros((NB,NB,NB,NB))
OE = np.zeros((NB))
AOInt = np.zeros((NB, NB, NB, NB))
twoE = np.zeros((NB, NB, NB, NB))
O2 = O*2
V2 = V*2
IJKL = np.zeros((O2,O2,O2,O2))
ABCD = np.zeros((V2,V2,V2,V2))
IABC = np.zeros((O2,V2,V2,V2))
IJAB = np.zeros((O2,O2,V2,V2))
IJKA = np.zeros((O2,O2,O2,V2))
IAJB = np.zeros((O2,V2,O2,V2))

# Get 2e integrals
AOInt = get2e(molecule, AOInt)

# Change to spin orbital form
IJKL, ABCD, IABC, IJAB, IJKA, IAJB = conMO(O, V, NB, Coeff, AOInt, IJKL, ABCD, IABC, IJAB, IJKA, IAJB)

# Initialize T1 and T2
t1 = np.zeros((O2, V2))
t2 = np.zeros((O2, O2, V2, V2))
# Define Denominator Arrays
D1 = np.zeros((O2, V2))
D2 = np.zeros((O2, O2, V2, V2))
# Initial T2 Guess
for i in range(O2):
  for j in range(O2):
    den = Fock[i,i]+Fock[j,j]
    for a in range(V2):
      for b in range(V2):
        D2[i,j,a,b] = den-Fock[a+O2,a+O2]-Fock[b+O2,b+O2]
t2 = IJAB/D2        
# D1 denominator, D2 formed in previous loop
for a in range(V2):
  for i in range(O2):
    D1[i,a] = Fock[i,i]-Fock[a+O2,a+O2]

# Initialize more values for the main loop
E_Corr2 = 0
DiffE = 1
st = []
st1 = []
st2 = []
e = []
B = np.zeros((6,6))
Max = 5
N = 0
MaxIt = 70
total_time = 0

# CCSD T and E Loop
with open(f"{molecule}.out","w") as writer:
  writer.write("*******SOLVING CCSD T AMPLITUDE AND ENERGY*******\n___________________________________________________________________\n___________________________________________________________________\n\n")

# CCSD Convergence Loop
while DiffE > 1e-9 or DiffT1 > 1e-7 or DiffT2 > 1e-7 or t1RMSE > 1e-7 or t2RMSE > 1e-7 and N < MaxIt:
  E_Corr1 = E_Corr2
  # Calculate intermediates
  start = time.time()
  tau_tilde = Intermediate.tau_tildeEq(1, O, V, t1, t2)
  tau = Intermediate.tauEq(1, O, V, t1, t2)
  F_ae, F_mi, F_me, W_mnij, W_abef, W_mbej = Intermediate.intermediateEqs(1, O, V, Fock, t1, t2, IJKL, ABCD, IABC, IJAB, IAJB, IJKA, tau_tilde, tau)
  # Do t1 step
  t1_f = Amplitudes.t1Eq(1, O, Fock, t1, t2, IABC, IJKA, IAJB, F_ae, F_mi, F_me, D1)
  # Do t2 step
  t2_f = Amplitudes.t2Eq(1, t1, t2, IABC, IJAB, IJKA, IAJB, tau, F_ae, F_mi, F_me, W_mnij, W_abef, W_mbej, D2)
  # Calculate differences between updates
  DiffT1 = abs(np.max(t1_f-t1))
  DiffT2 = abs(np.max(t2_f-t2))
  t1RMSE = (np.sum(((t1_f-t1)**(2))/(np.size(t1))))**(1/2)
  t2RMSE = (np.sum(((t2_f-t2)**(2))/(np.size(t2))))**(1/2)
  t1 = np.copy(t1_f)
  t2 = np.copy(t2_f)
  # Do Energy step
  E_Corr2 = Energy.E_CCSD(O, Fock, t1, IJAB, tau)
  DiffE = abs(E_Corr2-E_Corr1)
  # DIIS Extrapolation
  if diis_mode == "True":
    t1_ext, t2_ext, st, st1, st2, e, B, DIIS = Extrapolation.diis(t1, t2, st, st1, st2, e, B, O2, V2, Max, N) 
    t1 = np.copy(t1_ext)
    t2 = np.copy(t2_ext)
  else:
    DIIS = False
  # Time steps
  step = time.time()-start
  total_time += step
  # Update output
  with open(f"{molecule}.out","a") as writer:
    writer.write(f"Iteration {N}: E_corr(CCSD) {E_Corr2}\n")
    writer.write(f"DIIS Extrapolation Used: {DIIS}\n")
    writer.write(f"E(CCSD): {scfE+E_Corr2}\n")
    writer.write(f"Step Time: {step}\n\n")
  N+=1
  if N > MaxIt:
    with open(f"{molecule}.out","a") as writer:
      writer.write(f"Convergence Failed: Max Iterations Reached\n")
    break
with open(f"{molecule}.out", "a") as writer:
  writer.write("---------------------------------------------------------------------------------------\n")
  writer.write(f"Total Calculation Time: {total_time} seconds\n")
  writer.write("Successful termination\n")


