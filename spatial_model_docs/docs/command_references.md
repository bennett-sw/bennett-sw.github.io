## Detailed Function Reference




#### `f`
Computes the Euclidean distance between two vectors `a` and `b` of equal length.

**Arguments**  
- `a`, `b` (sequence of numbers or Sympy symbols): Coordinate lists of equal length.

**Returns**  
- (`sqrt` expression): Euclidean distance \(\sqrt{\sum_i (a_i - b_i)^2}\).

**Step‑by‑Step**  
1. Compute `sum((a[i] - b[i])**2 for i in range(len(a)))`.  
2. Wrap with `sqrt(...)` from Sympy for symbolic or numeric evaluation.

---

#### `find_intersections`
Solves symbolically for the intersection points of two circles (given as dicts with `'center'` and `'radius'`) and returns a list of real `(x,y)` solutions.

**Arguments**  
- `circle1dict`, `circle2dict` (`Dict[str,Any]`): Each has `'center': (x,y)` and `'radius'`.

**Returns**  
- (`List[(float,float)]`): Real intersection points of the two circles.

**Step‑by‑Step**  
1. Declare symbolic `x,y`.  
2. Unpack `x1,y1,r1` and `x2,y2,r2` from the dicts.  
3. Form circle equations:  
   - `(x-x1)^2 + (y-y1)^2 = r1^2`  
   - `(x-x2)^2 + (y-y2)^2 = r2^2`.  
4. `solve` the system for `(x,y)`.  
5. Filter solutions where both coordinates are real; call `evalf()`.  
6. Return as list of `(float, float)` pairs.

---

#### `squared_loss`
Given a candidate point `x` and an anchor `point`, computes \((\|x - point\| - observed\_dist)^2\).

**Arguments**  
- `x` (`array_like` of length 2): Candidate coordinates.  
- `point` (`array_like` of length 2): Anchor coordinates.  
- `observed_dist` (`float`): Noisy observed distance.

**Returns**  
- (`float`): \((\|x-point\| - observed\_dist)^2\).

**Step‑by‑Step**  
1. Compute `predicted_dist = np.linalg.norm(x - point)`.  
2. Return `(predicted_dist - observed_dist)**2`.

---

#### `total_loss`
Sum of `squared_loss` over the three reference anchors (USA, RUS, CHN).

**Arguments**  
- `x` (`array_like`): Estimated point \([x,y]\).  
- `USA`, `RUS`, `CHN` (`array_like`): Anchor coordinates.  
- `dist_to_*` (`float`): Observed distances to each anchor.

**Returns**  
- (`float`): Sum of squared losses to all three anchors.

**Step‑by‑Step**  
1. Compute squared losses for each anchor via `squared_loss`.  
2. Return their sum.

---

#### `call_solvers`
1. Defines symbolic objective = sum of squared errors  
2. Computes its partials w.r.t. `x,y` via Sympy  
3. Lambdifies and calls `scipy.optimize.root` to solve the first‑order conditions  
4. Returns the solution `(x_sol, y_sol)` and its total squared error.

**Arguments**  
- `x1_val,...,y3_val` (`float`): Anchor coordinates for three anchors.  
- `d1_obs_val, d2_obs_val, d3_obs_val` (`float`): Observed distances.  
- `starting_point` (`tuple` of 2 floats): Initial guess for \((x,y)\).

**Returns**  
- `x_sol, y_sol` (`np.ndarray` of length 2): Solution to FOCs.  
- `error` (`float`): Sum of squared distance errors at that solution.

**Step‑by‑Step**  
1. Define symbols `x,y, x1...d3_obs`.  
2. Define `d1_impl = sqrt((x-x1)^2+(y-y1)^2)`, etc.  
3. Build `objective_function = sum((d_impl - d_obs)**2)`.  
4. Compute `df_dx = diff(objective, x)` and similarly `df_dy`.  
5. `lambdify` these into `f_dx_numeric`, `f_dy_numeric`.  
6. Define `equations(vars, ...)` that returns `[f_dx, f_dy]`.  
7. Solve with `root(equations, starting_point, args=(...))`.  
8. If successful, unpack `x_sol, y_sol = result.x` and compute  
   \[
     est\_d_i = \sqrt{(x\_sol - x_i)^2 + (y\_sol - y_i)^2},\quad error = \sum (est\_d_i - d\_i\_obs)^2.
   \]  
9. On failure, return \([0,0]\) and `inf`.

---

#### `call_optimizers`
Runs three off‑the‑shelf optimizers (`minimize`, `differential_evolution`, `dual_annealing`) on `total_loss`, compares their errors, and returns the best coordinate.

**Arguments**  
- `usa, russia, china` (`array_like`): Three anchor coords.  
- `dist_to_*` (`float`): Observed distances.  
- `initial_point_for_estimation` (`tuple`): Starting guess.  
- `distances_dictionary` (`Dict[str,float]`): Observed `f_i`.  
- `anchors_dictionary` (`Dict[str,float]`): Anchor coords keyed by `'x_i','y_i'`.

**Returns**  
- `(est_x, est_y)` (`float, float`): Best coordinate among three optimizers.

**Step‑by‑Step**  
1. Run `minimize(total_loss, ...)` → `res1`.  
2. Compute its squared error `error1`.  
3. Run `differential_evolution(...)` → `res2`, compute `error2`.  
4. Run `dual_annealing(...)` → `res3`, compute `error3`.  
5. Print all errors and solutions.  
6. Select the solution with lowest error; return its `(x,y)`.

---

#### `estimate`
Implements a custom Newton–Raphson loop (up to 100 iterations) to solve for `(x,y)` by linearizing the vector of implied distances. Tracks the best iterate and returns it as a Pandas Series.

**Arguments**  
- `anchors_dictionary` (`Dict[str, float]`): Maps `'x_i'` and `'y_i'` to the known coordinates of the three anchors.  
- `distances_dictionary` (`Dict[str, float]`): Maps `'f_1'`, `'f_2'`, `'f_3'` to the observed distances from the unknown point to each anchor.  
- `initial_guesses` (`List[Tuple[float, float]]`): A list of candidate starting points \((x_0, y_0)\) for the Newton–Raphson iterations.  
- `error_threshold` (`float`, optional): If the squared‑error falls below this threshold during any iteration, the algorithm stops early (default 0.0001).

**Returns**  
- A `pd.Series` containing the best estimates `{'est_x': x_best, 'est_y': y_best}` after exploring all initial guesses.

**Step‑by‑Step Explanation**

1. **Symbolic setup**  
   a. Declare Sympy symbols `x, y`.  
   b. Assemble the vector \(q = [x, y]^T\).  
   c. For each anchor \(i=1,2,3\), let \(q_i = [x_i, y_i]^T\) via `symbols('x_i')`, `symbols('y_i')`.  
   d. Form the symbolic distance vector  
   \[
     f(q, q_i) = \sqrt{(x - x_i)^2 + (y - y_i)^2},
   \]  
   then substitute in `anchors_dictionary` to get a numeric‑symbolic expression.

2. **Jacobian computation**  
   a. Define `f_vector = [f(q, q_1), f(q, q_2), f(q, q_3)]^T`.  
   b. Compute the \(3\times2\) Jacobian \(D = \partial f\_vector / \partial [x,y]\).  
   c. Substitute anchor values so that \(D\) is symbolic only in \(x,y\).

3. **Initialize tracking variables**  
   - `best_error = +∞`  
   - `best_result = None`

4. **Loop over each `initial_guess`**  
   For each \((x_0,y_0)\) in `initial_guesses`:  
   a. Set `q_current = [x_0, y_0]`.  
   b. Track `best_error_iter = +∞` and `best_result_iter = None`.  
   c. Initialize counters:  
      - `iters_no_improvement = 0`  
      - `no_iter_prev = 0`

5. **Newton–Raphson iterations**  
   Repeat up to `max_iterations = 100`:  
   a. Evaluate `D_eval = D.subs({'x': q_current[0], 'y': q_current[1]})`.  
   b. Evaluate `f_eval = f_vector.subs({'x': q_current[0], 'y': q_current[1]})`.  
   c. Form the update matrix  
      \[
        H = (D\_eval^T I^{-1} D\_eval),\quad
        g = D\_eval^T I^{-1} (r - f\_eval),
      \]  
      where \(I\) is the \(3\times3\) identity and \(r\) is the observed‑distance vector.  
   d. Compute the Newton step  
      \[
        \Delta q = H^{-1} g,\quad
        q_{\text{new}} = q_{\text{current}} + \Delta q.
      \]  
   e. Update `q_current = q_new`.  
   f. Compute the squared‑error  
      \[
        \text{error} = \sum_{i=1}^3 \bigl(\|q_{\text{new}} - q_i\| - f_i\bigr)^2.
      \]  
   g. **Check improvement**  
      - If `error < best_error_iter`, set  
        `best_error_iter = error`,  
        `best_result_iter = q_new`,  
        `no_iter_prev = 0`.  
      - Else, increment `no_iter_prev` and if `no_iter_prev == 1`, also increment `iters_no_improvement`.  
   h. **Early break**  
      - If `iters_no_improvement >= 5`, break the loop.  
      - If `error < error_threshold`, break immediately.

6. **Select best from this guess**  
   - After iterations end, compare `best_error_iter` to the overall `best_error`.  
   - If `best_error_iter < best_error`, update  
     - `best_error = best_error_iter`  
     - `best_result = best_result_iter`.

7. **Return final estimate**  
   - Once all `initial_guesses` are processed, wrap `best_result` into a `pd.Series({'est_x': ..., 'est_y': ...})` and return it.

---

#### `region_intersection_info`
For three circles, builds a fine grid to detect:
- How many pairwise intersections are non‑empty  
- Whether all three overlap  
- Returns those counts plus the actual grid points of overlap and a dict of pairwise intersection points.

**Arguments**  
- `center1`, `center2`, `center3` (`Tuple[float, float]`): Coordinates of the three circle centers.  
- `radius1`, `radius2`, `radius3` (`float`): Radii of the three circles.  
- `resolution` (`int`, optional): Number of grid points per axis for overlap detection (default 1000).

**Returns**  
- `non_empty_pairwise` (`int`): Count of circle pairs with non‑empty overlap.  
- `non_empty_common` (`bool`): Whether all three circles share a common overlapping region.  
- `intersection_points` (`np.ndarray` of shape `(n,2)`): Grid points in the triple‑overlap region.  
- `intersections` (`Dict[str, Dict]`): For each pair key (`'circle1_circle2'`, etc.), a dict containing:  
  - `'intersection_points'` (`np.ndarray` of shape `(m,2)`)  
  - `'third_circle_center'` (`Tuple[float,float]`)

**Step‑by‑Step Explanation**  
1. Declare Sympy symbols `x, y`.  
2. Unpack numeric `x1, y1, …, x3, y3` and radii `r1, r2, r3`.  
3. Define boolean inequalities for being inside each circle: ineq1: (x−x1)^2+(y−y1)^2 ≤ r1^2, etc.  
4. Build 1D arrays `x_vals, y_vals` spanning each circle’s extent at length `resolution`.  
5. Create meshgrid `grid_x, grid_y`.  
6. Lambdify each inequality into numpy functions `f_ineq1`, `f_ineq2`, `f_ineq3`.  
7. Evaluate to boolean masks `region1`, `region2`, `region3`.  
8. For each pair, collect `intersection_points = np.column_stack((grid_x[mask], grid_y[mask]))` and store with the third center.  
9. Count `non_empty_pairwise` by checking sizes.  
10. Compute `common_intersection_region = region1 & region2 & region3`, then extract its points.  
11. Return the four outputs.

---

#### `distance_from_line`
Distance of a point `(x,y)` to the line through the global `(x1,y1)`–`(x2,y2)`.

**Arguments**  
- `point` (`Tuple[float, float]`): The `(x,y)` to project.

**Returns**  
- (`float`): Perpendicular distance from `point` to the line through global `x1,y1`–`x2,y2`.

**Step‑by‑Step Explanation**  
1. Unpack `x, y` from `point`.  
2. Compute numerator:  
   \[
     |(y2−y1)x − (x2−x1)y + x2⋅y1 − y2⋅x1|.
   \]  
3. Compute denominator \(\sqrt{(y2−y1)^2 + (x2−x1)^2}\).  
4. Return numerator/denominator.

---

#### `circle_constraint`
Equality constraint that forces `point` onto the perimeter of a given circle.

**Arguments**  
- `point` (`Tuple[float, float]`): `(x,y)` to test.  
- `hub_circle_center` (`Tuple[float, float]`): Center of the hub circle.  
- `hub_circle_radius` (`float`): Radius of the hub circle.

**Returns**  
- (`float`): Value of \((x−cx)^2 + (y−cy)^2 − radius^2\); zero on the circle.

**Step‑by‑Step Explanation**  
1. Unpack `x,y` and `cx, cy`.  
2. Compute \((x−cx)^2 + (y−cy)^2 − hub_circle_radius^2\).  
3. Return the result.

---

#### `find_best_point`
Given a “hub” circle and two other circle centers, finds the point on the hub’s perimeter closest to the line connecting the other two centers.

**Arguments**  
- `hub_circle_center` (`Tuple[float, float]`): Center of the circle on which to find the best point.  
- `hub_circle_radius` (`float`): Radius of that circle.  
- `third_circle_centers` (`List[Tuple[float, float]]`): The two other circle centers.

**Returns**  
- (`np.ndarray` of shape `(2,)`): The optimal perimeter point.

**Step‑by‑Step Explanation**  
1. Unpack `center`, `radius`, and declare globals `x1,y1,x2,y2` from `third_circle_centers`.  
2. Set `initial_guess = hub_circle_center + [radius, 0]`.  
3. Define `constraint = {'type':'eq','fun':circle_constraint,'args':(hub_circle_center, radius)}`.  
4. Call `minimize(distance_from_line, initial_guess, constraints=[constraint])`.  
5. Extract `optimal_point = result.x`.  
6. Return `optimal_point`.

---

#### `find_midpoint_outside_circles`
Computes midpoints of the two circle perimeters along the line of centers, then averages them to get an “external” midpoint.

**Arguments**  
- `circle_1`, `circle_2` (`Dict`): Each has `'center':(x,y)` and `'radius'`.

**Returns**  
- (`np.ndarray` of shape `(2,)`): Midpoint of the external segment between the two circle perimeters.

**Step‑by‑Step Explanation**  
1. Unpack `center1, radius1` and `center2, radius2`.  
2. Compute `line_vec = center2 − center1` and normalize to `unit_vec`.  
3. Define `intersection_point(center, radius, unit_vec) = center + unit_vec*radius`.  
4. Compute `point_on_circle1` with `+unit_vec` and `point_on_circle2` with `−unit_vec`.  
5. Average them: `(point_on_circle1 + point_on_circle2)/2`.  
6. Return the midpoint.

---

#### `find_key_for_center`
Looks up which circle name in a dict matches a target center coordinate.

**Arguments**  
- `circles_list` (`Dict[str, Dict]`): Maps circle names to attributes including `'center'`.  
- `target_center` (`Tuple[float, float]`): The center to look up.

**Returns**  
- (`str` or `None`): The circle name whose center matches, or `None`.

**Step‑by‑Step Explanation**  
1. Iterate `for circle_name, attributes in circles_list.items()`.  
2. If `attributes['center'] == target_center`, return `circle_name`.  
3. If no match, return `None`.

---

#### `generate_grid_points_within_square`
Uniformly samples `num_points` on a square grid centered at `center`, side length `side_length`.

**Arguments**  
- `center` (`Tuple[float, float]`): Square’s center.  
- `side_length` (`float`): Full side length.  
- `num_points` (`int`): Total points desired (will use `sqrt(num_points)` per axis).

**Returns**  
- (`np.ndarray` of shape `(num_points,2)`): All grid points.

**Step‑by‑Step Explanation**  
1. Compute `half_side = side_length/2`.  
2. Let `n = int(sqrt(num_points))`.  
3. Build `x = linspace(cx−half_side, cx+half_side, n)` and similarly `y`.  
4. Create meshgrid `xx, yy = meshgrid(x, y)`.  
5. Flatten and stack: `grid_points = vstack((xx.flatten(), yy.flatten())).T`.  
6. Return `grid_points`.

---

#### `find_initial_point`
Combines:
- Exact triple intersection centroid if it exists  
- Closest pairwise intersection otherwise  
- Centroid of pairwise intersection points  
- Midpoint‑outside if no overlaps  
- Optionally, adds a grid of 100 points for four special `(state, time)` cases.

**Arguments**  
- `anchors_dictionary` (`Dict[str, float]`): Keys `'x_i'`,`'y_i'` for i=1..3.  
- `distances_dictionary` (`Dict[str, float]`): Keys `'f_i'` for i=1..3.  
- `state` (`str`), `time` (`int`): Used for special grid‑addition cases.

**Returns**  
- (`List[Tuple[np.float64, np.float64]]`): Initial guess points.

**Step‑by‑Step Explanation**  
1. Unpack centers `(x_i,y_i)` and radii `f_i`.  
2. Build `circles = {'circle_i':{'center':..., 'radius':...}}`.  
3. Compute pairwise intersection lists via `find_intersections`.  
4. Flatten those into `points_of_intersection`.  
5. Call `region_intersection_info` → `(pairwise_count, common, intersection_points, intersections)`.  
6. **Case logic**:  
   - **common**: centroid of `intersection_points`.  
   - **pairwise_count==1**: closest pairwise point to the third center.  
   - **pairwise_count==2**: hub circle strategy via `find_best_point`.  
   - **pairwise_count==3 & not common**: centroid of three “corner” points.  
   - **pairwise_count==0**: centroid of three external midpoints.  
7. Build `initial_points_both = [initial_point] + points_of_intersection`.  
8. If `(state,time)` in four special cases, generate 100 grid points around mean of centers with `generate_grid_points_within_square`, filter `y≥0`, and extend.  
9. Convert all to `(np.float64, np.float64)` tuples and return.

---

#### `prep_russia_data`
Extracts the two observed distances for Russia and the two anchor coordinates for USA & CHN from a row.

**Arguments**  
- `df` (`pd.DataFrame`): One‑year slice containing `'country','f_1','f_2','x_1','y_1','x_2','y_2'`.

**Returns**  
- `distances_dictionary` (`Dict[str, float]`): Keys `'f_1','f_2'`.  
- `subs_dict_anchors_coords` (`Dict[str, float]`): Keys `'x_1','y_1','x_2','y_2'`.

**Step‑by‑Step Explanation**  
1. Filter `df` for `country=='RUS'`, select `['f_1','f_2']`, reset index, take row 0.  
2. Build `distances_dictionary[f_i] = value`.  
3. Extract anchor coords `x_values`, `y_values` similarly.  
4. Build `subs_dict_anchors_coords[f'x_i'] = x_values[i-1]`, etc.  
5. Return both dicts.

---

#### `region_intersection_info_russia`
Like `region_intersection_info` but for two circles, and computes the analytical solutions when they overlap.

**Arguments**  
- `center1`, `center2` (`Tuple[float, float]`): Two circle centers.  
- `radius1`, `radius2` (`float`): Their radii.  
- `resolution` (`int`, optional): Grid resolution.

**Returns**  
- `non_empty_pairwise` (`int`): 0 or 1.  
- `non_empty_common` (`bool`): True if regions overlap.  
- `intersection_points` (`np.ndarray`): Points in the overlap region.  
- `intersections` (`Dict`): As before, one key `'circle1_circle2'`.  
- `solutions` (`List[Tuple]`): Symbolic solve outputs for the two‑circle system.

**Step‑by‑Step Explanation**  
1. Declare symbols `x,y`.  
2. Unpack `x1,y1,x2,y2`.  
3. Define `ineq1, ineq2`.  
4. Build grid `grid_x, grid_y` at given `resolution`.  
5. Lambdify & evaluate masks.  
6. Form `intersections['circle1_circle2']` with masked points.  
7. Count `non_empty_pairwise`; compute `common_intersection_region`.  
8. Extract `intersection_points`.  
9. If `non_empty_common`, solve analytically the equations of the two circles.  
10. Return all five outputs.

---

#### `find_initial_point_russia`
Chooses one of the two analytic solutions (highest‑y) when they overlap, or else the midpoint‑outside.

**Arguments**  
- `anchors_dictionary` (`Dict[str, float]`): `'x_1','y_1','x_2','y_2'`.  
- `distances_dictionary` (`Dict[str, float]`): `'f_1','f_2'`.

**Returns**  
- (`Tuple[float, float]`): Chosen initial point.

**Step‑by‑Step Explanation**  
1. Unpack `center1, center2` and `radius1, radius2`.  
2. Build `circles` dict.  
3. Call `region_intersection_info_russia` → `(pairwise_count, common, _, intersections, solutions)`.  
4. If `pairwise_count==1`, pick the solution with max y from `solutions`.  
5. If `pairwise_count==0`, call `find_midpoint_outside_circles`.  
6. Return the chosen point.

---

#### `estimate_russia`
A two‑circle variant of `estimate`, using Newton–Raphson for up to 100 iterations (or an immediate return if the initial y=0).

**Arguments**  
- `anchors_dictionary` (`Dict[str, float]`): `'x_1','y_1','x_2','y_2'`.  
- `distances_dictionary` (`Dict[str, float]`): `'f_1','f_2'`.  
- `initial_guess` (`Tuple[float, float]`): Starting `(x_0,y_0)`.  
- `error_threshold` (`float`, optional): Early‑stop threshold.

**Returns**  
- A `pd.Series` with `{'est_x':…, 'est_y':…}`.

**Step‑by‑Step Explanation**  
1. If `initial_guess[1]==0`, return it immediately.  
2. Else, mirror the two‑circle Newton–Raphson in `estimate`:  
   - Symbolic setup with two anchors, Jacobian \(2×2\).  
   - Iterate up to 100 times, updating via  
     \(\Delta q = (D^T I^{-1}D)^{-1}D^T I^{-1}(r−f)\).  
   - Track and early‑stop on improvement or threshold.  
3. Return best result as `pd.Series`.

---

#### `build_data`
1. Renames Gallup disapproval columns to `f_1,f_2,f_3`  
2. Zeros out self‑disapproval for reference countries  
3. Drops missing data and merges with reference coordinates

**Arguments**  
- `gallup_data` (`pd.DataFrame`): Columns `['year','country','disapproval_china','disapproval_russia','disapproval_usa']`.  
- `reference_coords` (`pd.DataFrame`): Contains `['year','x_1','y_1','x_2','y_2','x_3','y_3']`.

**Returns**  
- (`pd.DataFrame`): Merged dataset ready for estimation.

**Step‑by‑Step Explanation**  
1. Rename columns to `f_1,f_2,f_3`.  
2. Set self‑disapproval to zero for `"USA"` and `"RUS"`.  
3. Drop rows with any NA.  
4. Merge on `'year'` with `reference_coords`.  
5. Return the merged DataFrame.

---

#### `get_distances`
Extracts observed distances `f_1, f_2, f_3` for a given country from a DataFrame row and returns them as a float64‑valued dict.

**Arguments**  
- `anchor_distances_dataframe` (`pd.DataFrame`): Columns include `country` and `f_1, f_2, f_3`.  
- `country` (`str`): ISO code of the focal country.

**Returns**  
- (`Dict[str, np.float64]`): Keys `"f_1"`, `"f_2"`, `"f_3"` mapped to observed distances.

**Step‑by‑Step**  
1. Filter rows where `'country' == country` and select columns `['f_1','f_2','f_3']`.  
2. Reset index, take the first row of those three columns.  
3. Build a Python dict `distances_dictionary` mapping `f_i` to the extracted value.  
4. Convert each value to `np.float64` and return the resulting dict.

---


#### `get_data`
Given a `(state, time)`, filters the full dataset, extracts anchor coordinates into `anchor_locations`, and returns it along with the observed distances dict.

**Arguments**  
- `state` (`str`): Focal country code.  
- `time` (`int`): Year.

**Returns**  
- `anchor_distances_dict` (`Dict[str,float]`)  
- `anchors_dict_floats` (`Dict[str,np.float64]`)

**Step‑by‑Step Explanation**  
1. Filter global `data` for `year==time`.  
2. Reset index, extract `x_i, y_i` into `anchors_dict`.  
3. Convert each to `np.float64`.  
4. For the requested `state`, call `get_distances` to get `anchor_distances_dict`.  
5. Return both dicts.

---

#### `likelihood_function`
Gaussian likelihood of a candidate `(x,y)` given noisy distances to three anchors, using a year‑specific `sigma` from `sigma_dict`.

**Arguments**  
- `x, y` (`float`): Candidate coordinates.  
- `observed_distances` (`Dict[str,np.float64]`): `'f_1','f_2','f_3'`.  
- `anchor_locations` (`Dict[str,np.float64]`): `'x_i','y_i'`.

**Returns**  
- (`float`): Joint Gaussian likelihood.

**Step‑by‑Step Explanation**  
1. Lookup global `sigma = sigma_dict[annum]`.  
2. Unpack `x1,y1,…,x3,y3` and `d1,d2,d3`.  
3. Compute true distances `r_i = sqrt((x−x_i)^2+(y−y_i)^2)`.  
4. Return  
   \[
   \bigl(1/\sqrt{2\pi\sigma^2}\bigr)^3 \exp\bigl(-\sum (d_i−r_i)^2/(2\sigma^2)\bigr).
   \]

---

#### `int_of_likelihood`
Numerically integrates `likelihood_function` over a fixed square `[-1.5,1.5]^2`.

**Arguments**  
- `anchor_distances`, `anchor_placement` (`Dict`): Passed to `likelihood_function`.  
- `bounds` (ignored in code).

**Returns**  
- (`float`): Integrated likelihood over \([-1.5,1.5]^2\).

**Step‑by‑Step Explanation**  
1. Define `integrand(x,y)` calling `likelihood_function`.  
2. Hard‑code `x_min,x_max,y_min,y_max = -1.5,1.5`.  
3. Call `integrate.dblquad(integrand, x_min, x_max, λx→y_min, λx→y_max)`.  
4. Print and return the integral.

---

#### `bounded_likelihood`
Normalizes the likelihood by dividing by its integral, yielding a proper density.

**Arguments**  
- `x, y` (`float`)  
- `anchor_distances`, `anchor_placement` (`Dict`)  
- `total` (`float`): Normalization constant.

**Returns**  
- (`float`): `likelihood_function(x,y,…)/total`.

**Step‑by‑Step Explanation**  
1. Call `likelihood_function(...)`.  
2. Divide by `total`.  
3. Return the result.

---

#### `quantize_distribution_kmeans`

Quantizes a 2D distribution by clustering with K‑means.

**Arguments**  
- `samples` (`np.ndarray` of shape `(n_samples, 2)`):  The full set of 2D points.  
- `n_clusters` (`int`):  Number of clusters (centroids) to find.  
- `random_state` (`int`):  Seed for reproducibility of the K‑means algorithm.

**Returns**  
- `centers` (`np.ndarray` of shape `(n_clusters, 2)`):  The coordinates of each cluster center.  
- `weights` (`np.ndarray` of shape `(n_clusters,)`):  The fraction of original samples assigned to each cluster (sums to 1).

**Step‑by‑Step**  
1. **Fit K‑means**  
   ```python
   kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
   labels = kmeans.fit_predict(samples)
   centers = kmeans.cluster_centers_
   ```  
2. **Count assignments**  
   ```python
   counts = np.bincount(labels, minlength=n_clusters)
   ```  
3. **Compute weights**  
   ```python
   weights = counts.astype(float) / counts.sum()
   ```  
4. **Return** the `centers` and their corresponding `weights`.

---

#### `class LikelihoodEstimator`

Wraps the original likelihood logic so we only load the anchor data (and therefore the relevant likelihood) once per country‑year.

##### `__init__(anchor_distances, anchor_placement)`

**Arguments**  
- `anchor_distances` (`Dict[str, float]`): Maps `"f_1","f_2","f_3"` to the observed noisy distances.  
- `anchor_placement` (`Dict[str, float]`): Maps `"x_1","y_1","x_2","y_2","x_3","y_3"` to the known anchor coordinates.

**Behavior**  
Stores `self.observed` and `self.anchors` for repeated pdf evaluations.

##### `pdf(x, y)`

Computes the joint Gaussian likelihood at the point `(x,y)` given the stored anchors and distances.

**Step‑by‑Step**  
1. Lookup `sigma = sigma_dict[annum]`.  
2. Unpack  
   ```python
   x1, y1 = self.anchors['x_1'], self.anchors['y_1']
   x2, y2 = self.anchors['x_2'], self.anchors['y_2']
   x3, y3 = self.anchors['x_3'], self.anchors['y_3']
   d1, d2, d3 = self.observed['f_1'], self.observed['f_2'], self.observed['f_3']
   ```  
3. Compute true distances  
   ```python
   r1 = sqrt((x - x1)**2 + (y - y1)**2)
   r2 = sqrt((x - x2)**2 + (y - y2)**2)
   r3 = sqrt((x - x3)**2 + (y - y3)**2)
   ```  
4. Form the normalization constant and exponent  
   ```python
   norm     = (1/√(2πσ²))**3
   exponent = -((d1-r1)**2 + (d2-r2)**2 + (d3-r3)**2) / (2σ²)
   ```  
5. Return `norm * exp(exponent)`.

---

#### `empirical_distribution`

Builds a quantized approximation of the posterior “point” distribution for a given country‑year.

**Arguments**  
- `country` (`str`): ISO code of the target country.  
- `year` (`int`):  Year of the data.  
- `sample_size` (`int`): Number of raw rejection‑samples to draw.

**Returns**  
- `samples_quantized` (`np.ndarray` of shape `(n_clusters, 2)`):  The K‑means centroids (representative points).  
- `marginal_weighted` (`np.ndarray` of shape `(n_clusters,)`):  Their weights (sum to 1).

**Step‑by‑Step**  
1. **Load anchor info**  
   ```python
   anchor_distances, anchor_placement = get_data(country, year)
   le = LikelihoodEstimator(anchor_distances, anchor_placement)
   ```  
2. **Compute normalization**  
   ```python
   integral = int_of_likelihood(anchor_distances, anchor_placement, bounds)
   ```  
3. **Find PDF maximum**  
   ```python
   def custom_pdf(x,y): return le.pdf(x,y)/integral
   def neg_pdf(xy):    return -custom_pdf(*xy)
   M = -differential_evolution(neg_pdf, bounds).fun
   ```  
4. **Batch rejection sampling**  
   ```python
   def rejection_sampling_batch(pdf, bounds, M, N, batch=10000):
       …
       # draw batches of size batch, accept where U ≤ pdf(X,Y), stack until N
       return samples[:N]
   samples = rejection_sampling_batch(custom_pdf, bounds, M, sample_size)
   ```  
5. **Quantize** (non‑anchors)  
   ```python
   if country not in ['RUS','CHN','USA']:
       samples_quantized, marginal_weighted = quantize_distribution_kmeans(samples, 1000)
   else:
       # return the single known anchor point with weight [1]
       …
   ```  
6. **Return** the `(samples_quantized, marginal_weighted)`.



---


#### `get_info_for_row(task)`
Loads or computes the empirical distribution for a single country‐year and saves it to disk as a pickle.

**Arguments**  
- `task` (`Tuple[str, int]`): A 2‑tuple `(country, year)` identifying which distribution to compute.

**Returns**  
- `None`

**Step‑by‑Step**  
1. Unpack `country, year = task`.  
2. Call `empirical_distribution(country, year, 100000)` → `(samples, marginals)`.  
3. Package into `pdf_dict = {'samples': samples, 'marginals': marginals}`.  
4. Open a pickle file at the designated path `…/{country}_{year}_data.pkl`.  
5. `pickle.dump(pdf_dict, f)` to save.  

---

#### `get_info_for_all_rows`
Parallelizes `get_info_for_row` over a DataFrame of country‑year pairs.

**Arguments**  
- `df` (`pd.DataFrame`): Contains columns `['country','year']`.  
- `n_processes` (`int`, optional): Number of worker processes (default 6).

**Returns**  
- None

**Step‑by‑Step**  
1. Extract just the (country, year) pairs as a list of tuples:
1. Create a `Pool(processes=n_processes)`.  
2. Call `pool.map(get_info_for_row, [row for _, row in df.iterrows()])`.  
3. Close the pool and wait for completion.

---

#### `wasserstein_pot_weighted`
Computes the \(l\)-Wasserstein distance between two weighted point clouds using POT.

**Arguments**  
- `sample1`, `sample2` (`np.ndarray` of shape `(n,2)`): Point coordinate arrays.  
- `pdf_sample1`, `pdf_sample2` (`array_like`): Weight vectors summing to ≥0.  
- `l` (`int`): The order of the distance (e.g. 1 or 2).

**Returns**  
- (`float`): The \(l\)-Wasserstein distance.

**Step‑by‑Step**  
1. Normalize weights `a = pdf_sample1/ sum(pdf_sample1)` and `b = pdf_sample2/ sum(pdf_sample2)`.  
2. Compute cost matrix `M = ot.dist(sample1, sample2, metric='minkowski', p=l)`.  
3. Call `ot.emd2(a, b, M, numItermax=300000, numThreads=6)` to get the minimal cost.  
4. Return `res**(1/l)`.

---

#### `load_pickled_objects`
Loads two pickle files for a given country‑year pair (and optional method suffix).

**Arguments**  
- `directory` (`str`): Folder containing the pickle files.  
- `country1`, `country2` (`str`): ISO codes.  
- `year` (`int`): Year.  
- `method` (`int`): Suffix selector (1 → no suffix, else `_{method}`).

**Returns**  
- (`Dict[str,Any]`): Mapping keys `"COUNTRY-year"` to the loaded dicts.

**Step‑by‑Step**  
1. If `method == 1`, set `suffix = ''`, else `suffix = str(method)`.  
2. For each `country` in `(country1, country2)`:  
   a. Build `filepath = f"{directory}/{country}_{year}_data{suffix}.pkl"`.  
   b. Open and `pickle.load` into `obj`.  
   c. Store under key `f"{country}-{year}"`.  
3. Return the combined dict.

---

#### `compute_wasserstein_distances`
Computes pairwise Wasserstein distances across all rows in a DataFrame.

**Arguments**  
- `df` (`pd.DataFrame`): Contains `['country','year']`.  
- `p` (`int`): Distance order.  
- `method` (`int`): Suffix selector for pickle loading.

**Returns**  
- (`pd.DataFrame`): Square matrix of distances, indexed and columned by `df.country`.

**Step‑by‑Step**  
1. Let `n = len(df)`, initialize `distance_matrix = np.zeros((n,n))`.  
2. Loop `i` from `0` to `n-1`, `j` from `i` to `n-1`:  
   a. Extract `(country_i, year_i)` and `(country_j, year_j)`.  
   b. Call `load_pickled_objects(dir, country_i, country_j, year_i, method)` → `data_dict`.  
   c. If both keys exist in `data_dict`, extract `(samples_i, marginals_i)` and `(samples_j, marginals_j)`.  
   d. Compute `d = wasserstein_pot_weighted(samples_i, samples_j, marginals_i, marginals_j, p)`.  
   e. Set `distance_matrix[i,j] = distance_matrix[j,i] = d`.  
   f. Else set both entries to `np.nan`.  
3. Wrap into `pd.DataFrame(distance_matrix, index=df.country, columns=df.country)`.  
4. Return the DataFrame.

---

#### `unpack_distance_matrix`
Converts the upper triangle of a square distance DataFrame into long‑form.

**Arguments**  
- `distance_df` (`pd.DataFrame`): Square matrix with countries as index and columns.  
- `p` (`int`): The distance order, used in the column name.

**Returns**  
- (`pd.DataFrame`): Long‑form table with columns `['country 1','country 2','distance_w_p']`.

**Step‑by‑Step**  
1. Initialize empty list `rows`.  
2. For each `i` in `0..n-1`, `j` in `i+1..n-1`:  
   a. Let `c1 = distance_df.index[i]`, `c2 = distance_df.columns[j]`.  
   b. Let `d = distance_df.iloc[i,j]`.  
   c. Append `[c1, c2, d]` to `rows`.  
3. Create `pd.DataFrame(rows, columns=['country 1','country 2', f'distance_w_{p}'])`.  
4. Return it.  
