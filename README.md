"""
(i)   optimize initial-state basis (3-body: helion)
"""
python3 ./NextToNewestGeneration.py <out_DIR> <nbr_of_processorcores> -1 1

"""
(ii)  optimize final-state basis (3-body + E1 => J=0.5,1.5)
      +press enter to store the results (in the same folder as those of step (i)
"""
python3 ./NextToNewestGeneration.py <out_DIR> <nbr_of_processorcores> <idx_of_first_basis-scenario> <idx_of_last_basis-scenario>

"""
(iii) properly couple the vectors and store everything in 
      <out_DIR>/results
      ecce: for this step all that was written in /tmp in (i-ii) is not needed
"""
python3 ./A3_lit_M.py <out_DIR> <nbr_of_processorcores> <idx_of_first_basis-scenario> <idx_of_last_basis-scenario>

"""
(iv)  in helion_E1_Fj.nb set 
      resultsDir = <out_DIR>/results
      I_n = {0.5,1.5} if both channels were considered in (i-iii)
"""
run helion_E1_Fj.nb