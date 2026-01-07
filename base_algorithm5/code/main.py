import numpy as np
import casadi as ca

def solve_mpc(v0, a0, t1, d1, t2=None, d2=None):
    # Define optimization parameters
    T = 20  # Time steps
    dt = 0.1  # Time step length
    N = T + 1  # Total number of nodes including the initial one

    # Define optimization variables
    aa = ca.SX.sym('aa', T)  # Jerk

    # Define initial state parameters
    P = ca.SX.sym('P', 3)  # Initial parameters: [a0, v0, s0]

    s = ca.SX.sym('s', N)  # position
    v = ca.SX.sym('v', N)  # velocity
    a = ca.SX.sym('a', N)  # acceleration

    # Initialize states
    a[0] = P[0]
    v[0] = P[1]
    s[0] = P[2]

    # Dynamics equations for states based on initial conditions and optimization variable
    for i in range(T):
        a[i + 1] = a[i] + aa[i] * dt
        v[i + 1] = v[i] + a[i] * dt
        s[i + 1] = s[i] + v[i] * dt + 0.5 * a[i] * dt ** 2

    # Constraints
    g = []
    lbg, ubg = [], []
    for i in range(1, N):
        g.append(v[i])  # 速度约束
        g.append(a[i])  # 加速度约束
        lbg.append(0)  # 速度下界
        ubg.append(15)  # 速度上界
        lbg.append(-5)  # 加速度下界
        ubg.append(3)  # 加速度上界

    lbx, ubx = [], []
    for _ in range(T):
        lbx.append(-30)
        ubx.append(30)

        # Objective function components
    cost_position = 0
    cost_velocity_deviation = ca.sum1(ca.sumsqr(v - 8))  # Deviation from target speed
    cost_jerk = ca.sum1(ca.sumsqr(aa))  # Minimize jerk

    # Handle terminal constraints and costs
    if t1 > T * dt:
        s_t1 = s[T] + (t1 - T * dt) * v[T] + 0.5 * a[T] * (t1 - T * dt) ** 2
        cost_position += (s_t1 - d1) ** 2
    else:
        cost_position += (s[int(t1 / dt + 0.5)] - d1) ** 2

    if t2 is not None:
        if t2 > T * dt:
            s_t2 = s[T] + (t2 - T * dt) * v[T] + 0.5 * a[T] * (t2 - T * dt) ** 2
            cost_position += (s_t2 - d2) ** 2
        else:
            cost_position += (s[int(t2 / dt + 0.5)] - d2) ** 2

    obj = cost_position + cost_jerk * 0.1 + 0.02 * cost_velocity_deviation

    # Create optimization problem
    nlp = {'x': ca.vertcat(aa), 'f': obj, 'g': ca.vertcat(*g), 'p': P}

    # Solver options
    opts_setting = {'ipopt.max_iter': 1000, 'ipopt.print_level': 0, 'print_time': 0,
                    'ipopt.acceptable_tol': 1e-5, 'ipopt.acceptable_obj_change_tol': 1e-4}

    # Create and solve the NLPSolver
    solver = ca.nlpsol('solver', 'ipopt', nlp, opts_setting)

    # Initial guess for 'aa'
    x0 = np.zeros(T)

    # Solve the optimization problem
    res = solver(x0=x0, p=np.array([min(max(a0, -5), 3), v0, 0]), lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
    return res


res = solve_mpc(5, 0, 1, 6, 2, 9)

