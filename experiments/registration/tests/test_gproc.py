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




rcc_signal = [ 192.02541542,   87.93687248,  181.25763512,  116.65095329,  184.84689522,
  145.3650341,   183.05226517,   77.16909218,   78.96372223,  118.44558334,
  109.47243309,  188.43615532,   75.37446213,   64.60668182,   89.73150253,
  181.25763512,  138.1865139,   183.05226517,  145.3650341,   150.74892426,
   86.14224243,  105.88317299,  147.15966415,  109.47243309,   62.81205177,
  132.80262375,  170.48985481,   93.32076263,  143.57040405,  166.90059471,
  136.39188385,  195.61467552,  214.12439346,   57.09983826,  164.16203499,
   87.43412733,  187.35884428,  167.73077488,  171.29951477,  223.04624319,
  137.39648581,  224.83061314,   99.92471695,   74.94353771,  103.49345684,
  148.10270548,   60.66857815,  157.02455521,  146.31833553,  180.2213645,
  215.90876341,   83.86538744,  165.94640493,  139.18085575,  173.08388472,
  135.61211586,   83.86538744,  146.31833553,   94.57160711,   91.00286722,
  167.73077488,  119.55278635,  155.24018526,   64.23731804,  166.9966352,
   94.15767729,  181.20911479,   76.3920778,   158.11383545,  193.64503443,
  197.19815433,  202.52783418,  218.51687372,   76.3920778,    90.60455739,
   88.82799745,  243.388713,     74.61551785,  188.31535459,   53.29679847,
  197.19815433,  111.92327678,  152.78415561,  211.41063392,   99.48735714,
  147.45447576,   71.06239796,  117.25295663,  222.06999362,  151.00759566,
   74.61551785,  104.81703699,  133.24199617,   99.48735714,  216.74031377,
  214.96375382,  106.156932,     70.17153132,  212.31386399,   62.97445118,
  246.49999464,   70.17153132,   82.76642156,   77.36861145,  192.52189362,
  172.72992325,  134.94525254,   80.96715152,  165.53284311,  192.52189362,
   57.57664108,  115.15328217,  221.31021416,  188.92335355,  158.33576298,
   61.17518115,   79.16788149,  205.11678386,  181.72627342,  172.72992325,
  212.31386399,  152.93795288,  235.70437443,   86.36496162,  129.54744244,
  136.74452257,  165.53284311,  140.34306264]
def visual_test_spherical_poly_conditional():
    import experiments.registration.gproc as gp
    import numpy.linalg as npl
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    import numpy as np

    #B = np.loadtxt('data/B.txt')[1:,:3]
    B = np.loadtxt('Ramon_dwi.bvecs')[:,:3]

    noise = np.array([21, 19, 26, 16, 20, 20, 26, 14, 24,  8,
                      37,  9, 39, 29, 17,  7, 13, 23, 12, 55, 40,
                      19,  6,  5, 20, 23, 31, 25, 14, 19, 22, 14], dtype=np.float64)
    #signal = np.array([186, 107, 167, 250, 170, 135, 93, 250, 138,  95, 169,
    #                   207, 177, 160, 247, 188, 116, 235, 199, 192, 153, 237,
    #                   176, 115, 228, 200, 90, 157, 194, 216, 94, 128], dtype=np.float64)
    signal = np.array(rcc_signal, dtype=np.float64)
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






