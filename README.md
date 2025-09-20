CCSD Energy
----------------

This is a program which calculates the CCSD energy for a molecule using the results of a 
Hartree Fock calculation from the GAUSSIAN suite of programs.
The open-source software Gauopen was then used to extract the needed quantities and write them to text files.
Specifically, it reads in the 2 electron integrals, molecular orbital coefficients, and oribital energies.
These text files are located in the directories <molecule>_txts/
It then uses these to compute the correlation energy at CCSD level of theory.
I have included the necessary files for the molecules hydrogen (H2), lithium (Li2), and water (H2O).
These are small systems utilizing minimal basis sets (less accurate descriptions of our orbitals) to keep the computational cost low for the user.

This code is not fully optimized for performance, but instead it focuses on readability to help the 
user better understand the equations and how to implement them into code. The full equations are shown in the pdf document included in this folder. Descriptions for each part of the program are included below.
______________________________________________________________________________________________________________________________________________________
runccsd.py contains the main loop and calls functions from the other.py files

ccsdEqs.py contains the working CCSD equations

readInp.py reads the input information from the text files, and transforms the 2e- integrals from atomic orbital basis to molecular orbital basis

diis.py performs the extrapolation to reduce the number of iterations needed to reach convergence
______________________________________________________________________________________________________________________________________________________

To run the code, use:
	"python runccsd.py CCSD.inp"
You can use the CCSD.inp input file to enter which molecule to use and whether or not to include the DIIS extrapolation.

This will generate a text file <molecule>.out, which prints the correlation energy at each step, along with timestamps and information about the DIIS extrapolation.
The user can verify that they are receiving the correct result by comparing with the reference calculations in the ref-outputs/ directory.
 

