
## Overview of File Structure

#### 1. **Imports**  
Standard libraries (`numpy`, `pandas`, `sympy`, `scipy.optimize`, etc.) and the POT optimal‑transport library.

#### 2. **Data‑Preparation Helpers**  
   - [ensure_pos_distance](command_references.md#ensure_pos_distance)  
   - [get_distances](command_references.md#get_distances)

#### 3. **Distance Utilities**  
   - [f](command_references.md#f)  
   - [find_intersections](command_references.md#find_intersections)

#### 4. **Loss & Solver Wrappers**  
   - [squared_loss](command_references.md#squared_loss)  
   - [total_loss](command_references.md#total_loss)  
   - [call_solvers](command_references.md#call_solvers)  
   - [call_optimizers](command_references.md#call_optimizers)  
   - [estimate](command_references.md#estimate)

#### 5. **Initial‑Point Generation (Three‑Circle Case)**  
   - [region_intersection_info](command_references.md#region_intersection_info)  
   - [distance_from_line](command_references.md#distance_from_line)  
   - [circle_constraint](command_references.md#circle_constraint)  
   - [find_best_point](command_references.md#find_best_point)  
   - [find_midpoint_outside_circles](command_references.md#find_midpoint_outside_circles)  
   - [find_key_for_center](command_references.md#find_key_for_center)  
   - [generate_grid_points_within_square](command_references.md#generate_grid_points_within_square)  
   - [find_initial_point](command_references.md#find_initial_point)

#### 6. **Russia‑Specific Variants**  
   - [prep_russia_data](command_references.md#prep_russia_data)  
   - [region_intersection_info_russia](command_references.md#region_intersection_info_russia)  
   - [find_initial_point_russia](command_references.md#find_initial_point_russia)  
   - [estimate_russia](command_references.md#estimate_russia)

#### 7. **Building the Main Dataset**  
   - [build_data](command_references.md#build_data)

#### 8. **Anchor‑Distance & Likelihood**  
   - [get_data](command_references.md#get_data)  
   - [likelihood_function](command_references.md#likelihood_function)  
   - [int_of_likelihood](command_references.md#int_of_likelihood)  
   - [bounded_likelihood](command_references.md#bounded_likelihood)

#### 9. **Distribution Quantization & Sampling**  
   - [quantize_distribution](command_references.md#quantize_distribution)  
   - [empirical_distribution](command_references.md#empirical_distribution)

#### 10. **Parallel Processing & Export**  
   - [get_info_for_row](command_references.md#get_info_for_row)  
   - [get_info_for_all_rows](command_references.md#get_info_for_all_rows)

#### 11. **Wasserstein Computation**  
   - [wasserstein_pot_weighted](command_references.md#wasserstein_pot_weighted)  
   - [load_pickled_objects](command_references.md#load_pickled_objects)  
   - [compute_wasserstein_distances_v2](command_references.md#compute_wasserstein_distances_v2)  
   - [unpack_distance_matrix](command_references.md#unpack_distance_matrix)

#### 12. **Main Execution**  
    Under `if __name__ == '__main__':`, iterates over years, computes and saves distances.

---

