# MSc-Thesis-Code-k-independence-Inertial-Bound
The code used for my MSc Thesis containing a code for the new MILP and a code for the old MILP for the inertia-type bound found in Abiad, A., Coutinho, G., &amp; Fiol, M. A. (2019), "On the k-independence number of graphs", Discrete Applied Mathematics, 242, 26–35. doi: 10.1016/j.dam.2017 .12.015

The code for the new model contains a method to print the complete MILP and the graph. It uses sage to build graph. The first method just generate a random graph but this can obviously be changed to run any graph needed. It also uses gurobi to solve the MILP.

OldModelMILP contains the old model, NewModelMILP contains the one proposed in the thesis and the other files contains the codes, the list of graphs and results used to compare the running times of both models on 50 graphs.
