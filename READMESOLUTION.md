Method
Localization: Uses BBox data to find the traffic light and DBSCAN clustering to identify the golf cart, barrels, barriers, and pedestrians.

Coordinate Mapping: Sets the Traffic Light as the origin (0,0). All movement is calculated relative to this fixed point.

Color Detection: Analyzes RGB patches to classify the light state (Red/Yellow/Green).

Refinement: Applies Median and Gaussian filters to smooth jumpy sensor data and Cubic Interpolation to create fluid 100-frame trajectories.

2. Assumptions
Fixed Anchor: The traffic light is stationary and serves as the global reference point.

Geometric Heuristics: Objects are classified by size and height (e.g., pedestrians are tall/narrow; barrels are short/cylindrical).

Sensor Validity: Assumes depth data is accurate enough within 50 meters to form identifiable point clusters.

3. Results
Visual Output: Generates a .gif showing the synchronized movement of the Ego vehicle, golf cart, and pedestrians.

Map Features: Produces a static map (reconstruction_map.png) plotting consistent environmental obstacles like barrels and guardrails.

Dynamic UI: Includes a real-time status box in the animation that updates based on the detected traffic light color.

Unfortunately, the code was unable to detect the pedestrians correctly