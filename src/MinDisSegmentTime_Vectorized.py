import numpy as np
import jax
import jax.numpy as jnp


def MinDisSegmentPoint_Vectorized(p1, p2, p):
    '''
    Minimum distance between two line segments when the parametric parameter
    corresponds to time
    
    Inputs:
    p1 -> nxm point matrix of starting point of segment 1 [n points by m dimensions]
    p2 -> nxm point matrix of ending point of segment 1 [n points by m dimensions]
    p  -> point [1 x m]
    
    Outputs:
    dmin   -> minimum distance between line segments
    tout   -> parametric parameter 
    p1[s]  -> starting point of line segmentment for mninimum distance
    p2[s]  -> ending point of line segmentment for mninimum distance
    s      -> index in p1 and p2 that given the shortest distance from p
    
    Formulation:
    p_0 = p1 + t * (p2 - p1)
    
    min(p_0 - p)
    
    d**2 = (p1x-px + (p2x-p1x)*t)**2 + (p1y-py + (p2y-p1y)*t)**2 + ((p1z-pz + (p2z-p1z)*t))**2
    
    set derivative to 0 and solve for t
    
    '''

    # obtain vectors
    v = p1 - p
    u = p2 - p1
    
    # get parametric parameter from p = p1 + t*(p2-p1)
    t = -np.divide(np.sum(np.multiply(v, u), axis=1), np.sum(np.multiply(u, u), axis=1))
    
    # initialize distance vector
    dis = 0 * t.copy()

    # find logical conditions for where the shortest distance is between points or outside
    tm = t < 0
    tp = t > 0
    ti = np.logical_and(~tm, ~tp)

    # check conditions, calculate and store distances
    if(any(tm)):
        # tout = 1.
        dis[tm] = np.linalg.norm(p1[tm] - p, axis=1)
    if(any(tp)):
        # tout = 0.
        dis[tp] = np.linalg.norm(p2[tp] - p, axis=1)
    if(any(ti)):
        dis[ti] = np.linalg.norm(p1[ti] + np.multiply(t[ti].reshape((-1, 1)), u[ti]) - p, axis=1)
    
    # find single instances of minimum distance
    s = np.argmin(dis)
    dmin = dis[s]
    tout = t[s]

    return tout, dmin, p1[s], p2[s], s


def MinDisSegmentPoint_Vectorized_jax(p1, p2, p):
    '''
    Minimum distance between two line segments when the parametric parameter
    corresponds to time with JAX

    Inputs:
    p1 -> nxm point matrix of starting point of segment 1 [n points by m dimensions]
    p2 -> nxm point matrix of ending point of segment 1 [n points by m dimensions]
    p  -> point [1 x m]

    Outputs:
    dmin   -> minimum distance between line segments
    tout   -> parametric parameter
    p1[s]  -> starting point of line segmentment for mninimum distance
    p2[s]  -> ending point of line segmentment for mninimum distance
    s      -> index in p1 and p2 that given the shortest distance from p

    Formulation:
    p_0 = p1 + t * (p2 - p1)

    min(p_0 - p)

    d**2 = (p1x-px + (p2x-p1x)*t)**2 + (p1y-py + (p2y-p1y)*t)**2 + ((p1z-pz + (p2z-p1z)*t))**2

    set derivative to 0 and solve for t

    '''

    # vectors
    u = p2 - p1
    v = p1 - p

    # get parametric parameter from p = p1 + t*(p2-p1)
    uu = jnp.sum(u * u, axis=1)
    vu = jnp.sum(v * u, axis=1)

    t = -vu / uu

    t_clamped = jnp.clip(t, 0.0, 1.0)

    p_closest = p1 + t_clamped[:, None] * u

    dis = jnp.linalg.norm(p_closest - p, axis=1)

    s = jnp.argmin(dis)
    dmin = dis[s]
    tout = t[s]

    return tout, dmin, p1[s], p2[s], s
    
    
def main():
    # testing code

    p1 = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]])
    p2 = np.array([[1, 5, 0], [10, 1, 0], [0, 0, 1], [10.5, 0, 0]])

    p = np.array([90.45, 0, 0]).reshape((1, -1))

    t, d, v1, v2, _ = MinDisSegmentPoint_Vectorized(p1, p2, p)

    print(v1, v2)

    print(t, d)

    p1 = jnp.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]])
    p2 = jnp.array([[1, 5, 0], [10, 1, 0], [0, 0, 1], [10.5, 0, 0]])

    p = jnp.array([90.45, 0, 0]).reshape((1, -1))

    t, d, v1, v2, _ = MinDisSegmentPoint_Vectorized_jax(p1, p2, p)

    print(v1, v2)

    print(t, d)


if __name__ == '__main__':
    #  cProfile.run('main()')
    main()
