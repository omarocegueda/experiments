import experiments.registration.gproc as gp
import numpy.linalg as npl


def visual_test_sample_gp_prior():
    n = 20
    ell = 10.0
    nsamples = 5
    x = np.array(range(n), dtype=np.float64)
    sigma = gp.squared_exp(x, ell)
    sigma = np.array(sigma)
    eval, evec = npl.eigh(sigma)
    eval = np.sqrt(eval)
    L = evec.dot(diag(eval))
    C = npl.cholesky(sigma)

    figure()
    for i in range(nsamples):
        r = np.random.normal(size=n)
        s = L.dot(r)
        plot(s)

    figure()
    for i in range(nsamples):
        r = np.random.normal(size=n)
        s = C.dot(r)
        plot(s)


def visual_test_sample_gp_no_noise():
    n_in = 3
    x_in = np.array([3.5, 10.5, 12.5], dtype=np.float64)
    f_in = np.array([3.0, 10.0, 12.0], dtype=np.float64)

    n_out = 20
    ell = 10.0
    x_out = np.array(range(n_out), dtype=np.float64)

    mean_out, S_out = gp.squared_exponential_conditional(x_in, f_in, x_out, ell, 0.0)
    mean_out = np.array(mean_out)
    S_out = np.array(S_out)


    nsamples = 5
    eval, evec = npl.eigh(S_out)
    eval = np.sqrt(eval)
    L = evec.dot(diag(eval))
    C = npl.cholesky(S_out)

    figure()
    for i in range(nsamples):
        r = np.random.normal(size=n_out)
        s = L.dot(r)
        plot(s + mean_out)
        grid()

    figure()
    for i in range(nsamples):
        r = np.random.normal(size=n_out)
        s = C.dot(r)
        plot(s + mean_out)
        grid()


def visual_test_sample_gp_noise():
    x_in = np.array([3.5, 10.5, 12.5, 15.0, 20.0], dtype=np.float64)
    f_in = np.array([1.0, 10.0, 7.0, 0.0, 5.0], dtype=np.float64)
    n_in = x_in.shape[0]

    n_out = 20
    ell = .1
    sigma_noise = 0.1
    #x_out = np.array(range(n_out), dtype=np.float64)
    x_out = x_in.copy()
    n_out = x_out.shape[0]

    mean_out, S_out = gp.squared_exponential_conditional(x_in, f_in, x_out, ell, sigma_noise)
    mean_out = np.array(mean_out)
    S_out = np.array(S_out)

    nsamples = 1
    eval, evec = npl.eigh(S_out)
    eval = np.sqrt(eval)
    L = evec.dot(diag(eval))
    C = npl.cholesky(S_out)

    figure()
    plot(f_in)
    for i in range(nsamples):
        r = np.random.normal(size=n_out)
        r[:] = 0
        s = L.dot(r)
        plot(s + mean_out)
        grid()

    figure()
    plot(f_in)
    for i in range(nsamples):
        r = np.random.normal(size=n_out)
        r[:] = 0
        s = C.dot(r)
        plot(s + mean_out)
        grid()
        
        
def visual_test_spherical_poly_conditional():
    import experiments.registration.gproc as gp
    import numpy.linalg as npl
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    import numpy as np

    B = np.loadtxt('data/B.txt')[1:,:3]
    noise = np.array([21, 19, 26, 16, 20, 20, 26, 14, 24,  8,
                      37,  9, 39, 29, 17,  7, 13, 23, 12, 55, 40,
                      19,  6,  5, 20, 23, 31, 25, 14, 19, 22, 14], dtype=np.float64)
    signal = np.array([186, 107, 167, 250, 170, 135, 93, 250, 138,  95, 169,
                       207, 177, 160, 247, 188, 116, 235, 199, 192, 153, 237,
                       176, 115, 228, 200, 90, 157, 194, 216, 94, 128], dtype=np.float64)
    mean_signal = signal.mean()
    f_in = signal - mean_signal
    x_in = B.copy()
    sigmasq_signal = f_in.var()
    sigmasq_noise = noise.var()
    
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)

    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    x_out = np.array([x.reshape(-1),y.reshape(-1),z.reshape(-1)]).T
    mean_out, S_out = gp.spherical_poly_conditional(x_in, f_in, x_out, sigmasq_signal, sigmasq_noise)
    mean_out = np.array(mean_out)
    predicted = mean_out.reshape(x.shape)
    predicted += mean_signal
    x *= predicted
    y *= predicted
    z *= predicted
    fig = figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z,  rstride=4, cstride=4, color='b', alpha=0.2, shade=True)
    points = diag(signal).dot(x_in)
    ax.scatter(points[:,0].copy(), points[:,1].copy(), points[:,2].copy(), c='r', s=40)
    ax.scatter(-1*points[:,0].copy(), -1*points[:,1].copy(), -1*points[:,2].copy(), c='r', s=40)
    
    
    
    
    
    
