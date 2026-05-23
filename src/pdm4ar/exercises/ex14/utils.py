import math
from dataclasses import dataclass

import shapely
import numpy as np
from dg_commons import PlayerName
from dg_commons.sim import InitSimGlobalObservations
from numpydantic import NDArray
from pydantic import BaseModel
from scipy.sparse import csr_array as CSR
from scipy.stats import qmc
from scipy.spatial import KDTree as KDT
import scipy.sparse
from scipy.sparse.csgraph import shortest_path as scipy_shortest_path


class GlobalPlanMessage(BaseModel):
    per_agent_tasks: dict[str, list[tuple[str, str]]]
    global_agent_priorities: dict[str, int]
    per_agent_paths: dict[PlayerName, list[tuple[float, float]]]
    per_agent_flags: dict[PlayerName, list[int]]
    vertices: NDArray
    goals: int


@dataclass(frozen=True)
class Pdm4arAgentParams:
    param1: float = 10
    # Turn-in-PLace (TiP) CONTROLLER PARAMS
    angular_tol: float = 0.01  # Tolerance for angular position error when turning in place
    pos_tol: float = 0.05  # Tolerance for position to follow next checkpoint
    kp_pos: float = 20.0  # Kp parameter of P controller for position
    kp_lat: float = 2.0  # Kp parameter of P controller for lateral deviation
    kp_omega: float = 5.0  # Kp parameter of P controller for hea
    kp_omega_in_place: float = 2.0
    # Pure - Pursuit (PP) parameters
    window_r: int = 6  # number of sampled points from the path to check circle-line intersections
    look_ahead_d_start: float = 0.5  # look-ahead standard radius for PP controller

    collection_radius: float = 0.3  # Placeholder for goal collection points radius
    eps: float = 1e-2  # tolerance for checking point inclusion in in line-segment intersections
    collection_eps: float = collection_radius / 2  # tolerance for collection

    # ONLINE AVOIDANCE paramaters
    r_agent = 0.6
    obst_eps_buffer = 0.2
    r_priority: float = 2.5  # radius inside which obstacle avoidance is used to stop the lower priority robot
    r_fallback: float = 2  # radius inside which obstacle avoidance is used to reverse the lower priority robot
    r_bad: float = 1.5  # radius inside which obstacle avoidance is used to stop the higher priority robot
    vision_cone: float = (
        np.pi / 2
    )  # the angle from center in which agent detections count (only look for agents +- angle from direction of movement)
    sampling_eps_forward = 0.1  # sampling distance to the lookahead radius away from agent
    sampling_eps_backwards = 0.35  # sampling distance to the lookahead radius towards agent
    reversing_timesteps = 12  # number of timestep to reverse after dropping off all the goals


@dataclass
class RobotInitState:
    x: float
    y: float
    psi: float
    radius: float
    wheelbase: float
    wheelradius: float
    limits: object

params = Pdm4arAgentParams()


def to_pi_minus_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def sgn(x):
    return 1 if x >= 0 else -1


def switch_gears(switched: int | bool, omega_1: float, omega_2: float):
    omega1_temp = omega_1
    omega2_temp = omega_2
    if switched:
        omega_1_new = -omega2_temp
        omega_2_new = -omega1_temp
        return omega_1_new, omega_2_new
    else:
        return omega1_temp, omega2_temp


def wheel_speeds_to_target(
    target_pt: np.ndarray,
    current_pos: np.ndarray,
    current_heading: float,
    params: Pdm4arAgentParams,
    wheelbase: float,
    wheelradius: float,
    v_max: float,
) -> tuple[float, float]:
    p_to_t_vec = target_pt - current_pos
    theta = np.arctan2(p_to_t_vec[1], p_to_t_vec[0])
    delta = to_pi_minus_pi(theta - current_heading)
    dtheta = params.kp_omega * delta
    omega = (wheelbase / (2 * wheelradius)) * dtheta

    u = v_max / wheelradius
    u -= abs(omega)
    return u - omega, u + omega


def buffered_obstacle_tree(static_obstacles, buffer_radius: float) -> shapely.STRtree:
    obstacles = [obstacle.shape.buffer(buffer_radius) for obstacle in static_obstacles]
    return shapely.STRtree(obstacles)


def polygon_center_radius(polygon) -> tuple[np.ndarray, np.ndarray]:
    center = np.array([polygon.centroid.x, polygon.centroid.y])
    radius = np.linalg.norm(center[None, :] - polygon.exterior.coords[0], axis=1)
    return center, radius


def closest_point_in_distance_band(
    points: list[np.ndarray],
    current_pos: np.ndarray,
    target_distance: float,
    band: float,
) -> np.ndarray | None:
    if not points:
        return None

    pts = np.stack(points, axis=0)
    dists = np.linalg.norm(pts - current_pos, axis=1)
    mask = (dists >= target_distance - band) & (dists <= target_distance + band)
    candidate_idx = np.where(mask)[0]

    if len(candidate_idx) > 0:
        inside_dists = dists[candidate_idx]
        best_inside = candidate_idx[np.argmin(np.abs(inside_dists - target_distance))]
        return pts[best_inside]

    best_idx = np.argmin(np.abs(dists - target_distance))
    return pts[best_idx]


def task_indices(
    agents: list,
    goals: list,
    dropoffs: list,
) -> tuple[dict, set, dict, set, dict]:
    n_agents = len(agents)
    n_goals = len(goals)

    agent2idx = dict(zip(agents, range(n_agents)))
    agents_set = set(agents)
    goal2idx = dict(zip(goals, range(n_agents, n_agents + n_goals)))
    goals_set = set(goals)
    dropoff2idx = dict(zip(dropoffs, range(n_agents + n_goals, n_agents + n_goals + len(dropoffs))))
    return agent2idx, agents_set, goal2idx, goals_set, dropoff2idx


def route_distance(task_graph, agent, route) -> float:
    dist = 0
    prev = agent
    for stop in route:
        if prev == stop[0]:
            continue
        if task_graph.has_edge(prev, stop[0]):
            dist += task_graph[prev][stop[0]]["weight"]
        prev = stop[0]
    return dist


def route_label(route) -> str:
    path = []
    for stop in route:
        path.append(stop[0])
        path.append(stop[1])
    return " -> ".join(path)


def shortest_distance(graph, source_idx: int, target_idx: int) -> float:
    return scipy_shortest_path(graph, method="auto", directed=True, indices=[source_idx])[0, target_idx]


def priority(c_agent_goal: bool, o_agent_goal: bool, c_agent_priority: int, o_agent_priority) -> bool:
    """
    Docstring for priority

    :param carrying_goal: if the current agent is carrying a goal
    :type carrying_goal: bool
    :param agent_carrying_goal: if the observed agent is carrying a goal
    :type agent_carrying_goal: bool
    :param agent2priority: gives the priority ordering based on makespan
    :type agent2priority: dict
    :return: if the current agent has priority or not
    :rtype: bool
    """
    if c_agent_goal:
        # IF BOTH CARRING THE GOAL, CHECK THE PRIORITY LIST
        if o_agent_goal:
            priority = c_agent_priority > o_agent_priority
        else:
            # I HAVE PRIORITY IF THE OTHER AGENT IS NOT CARRYING THE GOAL
            priority = True
    else:
        # IF I AM NOT CARRYING A GOAL I DON'T HAVE PRIORITY IF THE OTHER ONE HAS IT
        if o_agent_goal:
            priority = False
        else:
            # CHECK PRIORITY LIST OTHERWISE
            priority = c_agent_priority > o_agent_priority

    return priority


def sample_points(current_pos, target_heading, vertex_tree, obstacle_tree, vertices) -> np.ndarray:
    """
    Finds a candidate point in a ring close to lookahead distance to back out from a possible collision

    :param current_pos: current position (x,y) of the robot avoiding the collision
    :param target_heading: direction the robot needs to take to avoid collision (opposite of other robot)
    :return: candidate point to target
    :rtype: ndarray[float, float]
    """
    # query the KDTREE of vertices

    neighbours = vertex_tree.query_ball_point(current_pos, r=params.look_ahead_d_start + params.sampling_eps_forward)
    pts = vertices[neighbours]
    dists = np.linalg.norm(pts - current_pos, axis=1)

    mask = (dists >= params.look_ahead_d_start - params.sampling_eps_backwards) & (
        dists <= params.look_ahead_d_start + params.sampling_eps_forward
    )
    filtered_idx = np.array(neighbours)[mask]
    filtered_pts = vertices[filtered_idx]
    # Now check if they are inside of an obstacle or inside of the colliding agent
    min_heading_diff = 2 * np.pi

    cand = None
    for v in filtered_pts:
        pt = shapely.geometry.Point(v).buffer(0.1)
        # check for collisions
        potential_collisions_idx = obstacle_tree.query(pt, predicate="intersects")

        if potential_collisions_idx.shape[0] == 0:
            pos_to_v = v - current_pos
            pos_to_v_heading = np.arctan2(pos_to_v[1], pos_to_v[0])
            psi_diff = to_pi_minus_pi(pos_to_v_heading - target_heading)
            # Among the feasible ones, pick the one closest to the target
            if abs(psi_diff) < min_heading_diff:
                min_heading_diff = abs(psi_diff)
                cand = v
    # Fallback return
    if cand is None:
        cand = current_pos
    return cand


def line_circle_inter(path, current_pos: np.ndarray, c_wpt_idx: int, look_ahead_d) -> list[np.ndarray]:
    """
    Computes candidate points on the path to follow based on the intersection between the circle around the robot and the segments

    :param path: High-level path to follow computed by other functions
    :param current_pos: Current position (x,y) of the robot
    """

    # FIND WAYPOINTS IN A WINDOW AROUND THE CURRENT POSITION
    indices = np.arange(c_wpt_idx, np.clip(c_wpt_idx + params.window_r, 0, len(path)).astype(int))

    # CREATE POINTS
    pts = [np.array(path[indices[i]]) for i in range(indices.shape[0])]
    pts = np.vstack(pts)  # Nx2
    # SHIFT POINTS FOR INTERSECTION CALCULATIONS
    pts[0] = current_pos[None, :]
    shifted_pts = pts - current_pos[None, :]
    solutions = []

    for i in range(indices.shape[0] - 1):
        dx = shifted_pts[i + 1, 0] - shifted_pts[i, 0]
        dy = shifted_pts[i + 1, 1] - shifted_pts[i, 1]
        dr_sq = dx**2 + dy**2
        D = shifted_pts[i, 0] * shifted_pts[i + 1, 1] - shifted_pts[i + 1, 0] * shifted_pts[i, 1]
        discriminant = (look_ahead_d**2) * (dr_sq) - D**2
        min_x = min(pts[i + 1, 0], pts[i, 0])
        min_y = min(pts[i + 1, 1], pts[i, 1])
        max_x = max(pts[i + 1, 0], pts[i, 0])
        max_y = max(pts[i + 1, 1], pts[i, 1])

        # if discriminant is >= 0, there exist solutions
        if discriminant >= 0:
            discr_sqrt = np.sqrt(discriminant)
            sol_x1 = (D * dy + sgn(dy) * dx * discr_sqrt) / dr_sq
            sol_x2 = (D * dy - sgn(dy) * dx * discr_sqrt) / dr_sq
            sol_y1 = (-D * dx + abs(dy) * discr_sqrt) / dr_sq
            sol_y2 = (-D * dx - abs(dy) * discr_sqrt) / dr_sq

            sol_1 = np.array([sol_x1, sol_y1])
            sol_1 += current_pos
            sol_2 = np.array([sol_x2, sol_y2])
            sol_2 += current_pos
            # FILTER OUT POINTS NOT BELONGING TO THE SEGMENT
            cond_1 = ((sol_1[0] <= max_x + params.eps) and (sol_1[0] >= min_x - params.eps)) and (
                (sol_1[1] <= max_y + params.eps) and (sol_1[1] >= min_y - params.eps)
            )
            cond_2 = ((sol_2[0] <= max_x + params.eps) and (sol_2[0] >= min_x - params.eps)) and (
                (sol_2[1] <= max_y + params.eps) and (sol_2[1] >= min_y - params.eps)
            )

            # Offset the system back to its original position
            if cond_1:
                solutions.append(sol_1)
            if cond_2:
                solutions.append(sol_2)

    return solutions


def valid_point(candidates: list[np.ndarray], current_pose: np.ndarray, last_target) -> np.ndarray:
    """
    Disambiguates between points found at the previous step
    :param points: candidate points
    :param current_pose: current pose (x,y,psi) of the robot
    :returns: (2,) position array of the target point
    """

    # IF THE LIST OF CANDIDATES IS EMPTY, RETURN the last target
    if not candidates:
        return last_target

    cands = np.vstack(candidates)  # (N,2)
    c_pos = current_pose[0:2]
    dir_vecs = cands - c_pos[None, :]
    heading = current_pose[2]
    heading_vec = np.array([np.cos(heading), np.sin(heading)])
    forward_mask = np.dot(dir_vecs, heading_vec) >= 0
    f_cands = cands[forward_mask]

    if len(f_cands) > 0:
        trgt = np.array(f_cands[0])
        return trgt
    else:
        if cands.shape[0] > 0:
            # return the last found point (which will be on the further segment)
            trgt = cands[-1, :]
            return trgt
        else:
            print("ERR")
            return last_target


def prm(
    agent_radius: float,
    boundaries: list,
    init_sim_obs: InitSimGlobalObservations,
    static_obstacles: list,
    eps_buffer: float,
    n_samples: int,
    sample_evenly: bool,
    log_data: bool,
) -> tuple[CSR, np.ndarray]:
    """
    Create a graph for the warehouse environment using PRM* (Probabalistic Road Map). The  graph
    returned is: an NxN undirected, spatially-weighted graph. Note first vertices are as follows:
    _a_ vertices corrispond to the agent starting locations,_g_ vertices corrispond to goal
    positions, and _d_ vertices corrispond to dropoff positions

    :param init_sim_obs: Global map
    :type init_sim_obs: InitSimGlobalObservations
    :return: Adjacency matrix
    :rtype: scipy.sparse.csr_array
    """

    # DONE: set to actual agent radius
    r_agent = agent_radius

    # create dense sequence of points in map
    minx, miny, maxx, maxy = boundaries
    l_bounds = [minx, miny]
    u_bounds = [maxx, maxy]
    if sample_evenly:
        # number of points
        n = n_samples

        # number of points in x and y such that x*y ≈ n
        nx = int(math.sqrt(n))
        ny = int(math.ceil(n / nx))

        # generate linearly spaced values
        xs = np.linspace(minx, maxx, nx)
        ys = np.linspace(miny, maxy, ny)

        # all grid points
        xx, yy = np.meshgrid(xs, ys)
        grid = np.vstack([xx.ravel(), yy.ravel()]).T

        # trim to exactly n samples
        sampled_pts = grid[:n]
    else:
        sampler = qmc.Halton(2)
        sampled_pts = qmc.scale(sampler.random(n_samples), l_bounds, u_bounds)

    # extract agent start points
    start_vertices = np.empty((0, 2))
    for start in list(init_sim_obs.initial_states.values()):
        center = np.array([start.x, start.y])
        start_vertices = np.row_stack((start_vertices, center))

    # extract task collection points
    collection_vertices = np.empty((0, 2))
    goal_ids = []
    for goal_id, collection in zip(list(init_sim_obs.shared_goals.keys()), list(init_sim_obs.shared_goals.values())):
        center = np.array([collection.polygon.centroid.x, collection.polygon.centroid.y])
        collection_vertices = np.row_stack((collection_vertices, center))
        goal_ids.append(goal_id)

    # extract task dropoff points
    dropoff_vertices = np.empty((0, 2))
    for dropoff in list(init_sim_obs.collection_points.values()):
        center = np.array([dropoff.polygon.centroid.x, dropoff.polygon.centroid.y])
        dropoff_vertices = np.row_stack((dropoff_vertices, center))

    # stack all defined points: agents, collection, dropoff
    defined_pts = np.row_stack((start_vertices, collection_vertices, dropoff_vertices))

    # create graph with vertices for all sampled points, agents, and task start and end points
    vertices = np.row_stack((defined_pts, sampled_pts))  # make sure shapes match here (should be n rows, 2 columns)
    n_vertices = vertices.shape[0]
    graph_dense = np.zeros((n_vertices, n_vertices))

    # create KDTree using only sampled points
    sampled_tree = KDT(sampled_pts)  # data should come in [n x m] with n=# pts, m=dimension

    # create KDTree with all points
    vertex_tree = KDT(vertices)

    # create buffered KDTree for shapely collisions
    obstacles = []
    for obstacle in static_obstacles:
        obstacles.append(obstacle.shape.buffer(r_agent + eps_buffer))
    obstacle_tree = shapely.STRtree(obstacles)

    # Query parameters for PRM*
    map_area = (maxx - minx) * (maxy - miny)
    gamma = 2.1 * np.sqrt(1 + 1 / 2) * np.sqrt(map_area / np.pi)
    r = gamma * np.sqrt((np.log(n_samples) / n_samples))

    # For every vertex, find nearest neighbors in given radius with 2-norm
    neighbors = vertex_tree.query_ball_tree(sampled_tree, r, 2)

    for u_idx, u_neighbors in enumerate(neighbors):

        if u_neighbors:

            u = vertices[u_idx]

            for v_idx_raw in u_neighbors:

                v = sampled_pts[v_idx_raw]
                path_seg = shapely.geometry.LineString((u, v))

                # check for collisions
                potential_collisions_idx = obstacle_tree.query(path_seg, predicate="intersects")

                if potential_collisions_idx.shape[0] == 0:
                    weight = np.linalg.norm(u - v)
                    v_idx = v_idx_raw + defined_pts.shape[0]
                    graph_dense[u_idx, v_idx] = weight
                    graph_dense[v_idx, u_idx] = weight
    if log_data:
        np.save("prm_vertices.npy", vertices)
        scipy.sparse.save_npz("prm_graph.npz", CSR(graph_dense))

    return CSR(graph_dense), vertices
