import numpy as np
import sys
from readInp import parsInp

class Extrapolation:
  def diis(t1_ext, t2_ext, st, st1, st2, e, B, O2, V2, Max, N_iter):
  #Form temporary arrays from t1 and t2 amplitudes
    ft1=list(t1_ext.flatten())
    ft2=list(t2_ext.flatten())
    st1.append(ft1)
    st2.append(ft2)
    ft = ft1 + ft2
    st.append(ft)
    if len(st) > 1:
      ev = np.array(st[len(st) - 1]) - np.array(st[len(st) - 2])
      e.append(ev)
    if len(e) > Max:
        del st1[0]
        del st2[0]
        del st[0]
        del e[0]
  #Start building B matrix
    if N_iter >= 5:
        B = np.zeros((Max + 1, Max + 1))
        for i in range(len(B)):
          for j in range(len(B)):
            if j == Max and i < Max:
              B[i][j] = 1
            if i == Max and j < Max:
              B[i][j] = 1
            if i < Max and j < Max:
              B[i][j] = np.dot(np.array(e[i]), np.array(e[j]))
    rhs = np.zeros(Max + 1)
    rhs[Max] = 1
    emat = B[:Max,:Max]
    ETest = np.max(abs(emat))
    t1d = np.zeros((len(st1[0])))
    t2d = np.zeros((len(st2[0])))
    DIIS = False
  #Turn on DIIS extrapolation every third iteration
    if ETest >= 1e-5 and N_iter % 3 == 0:
         DIIS = True
         csol = np.linalg.solve(B,rhs)
         csum = csol[0]+csol[1]+csol[2]+csol[3]+csol[4]
         for p in range(len(st) - 1):
             t1d += np.array(st1[p + 1]) * csol[p]
             t2d += np.array(st2[p + 1]) * csol[p]
         t1_ext = np.reshape(t1d,(O2,V2))
         t2_ext = np.reshape(t2d,(O2, O2, V2, V2))
    return t1_ext, t2_ext, st, st1, st2, e, B, DIIS
