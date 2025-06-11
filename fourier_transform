import numpy as np
import matplotlib.pyplot as plt


def fourier_trans(f, L, L_prime):
    N = int(np.ceil(L * L_prime))
    
    u = complex(0, 1) * 2 * np.pi * np.floor(N / 2)
    a = [np.exp(u * x/N) * f(L * (x/N - 0.5)) for x in range(N)]
    
    L_xi = np.arange(N) - np.floor(N/2)
    xi = L_xi / L
    Ff = np.power(-1, L_xi) / L_prime * np.fft.fft(a)
    
    return (xi, Ff)


def fourier_trans2(f, L, L_prime):
    """
    f: a 2d function
    L: a tuple (L1, L2)
    L_prime: a tuple (L1_prime, L2_prime)
    
    return: tuple ((xi1, xi2), Ff). xi1 is the first frequency grid, and xi2 
    is the second frequency grid. Ff is a 2d array containing values of the
    Fourier transform of f.
    """
    L1, L2 = L
    L1_prime, L2_prime = L_prime
    N1 = int(np.ceil(L1 * L1_prime))
    N2 = int(np.ceil(L2 * L2_prime))
    u1 = complex(0, 1) * 2 * np.pi * np.floor(N1 / 2)
    u2 = complex(0, 1) * 2 * np.pi * np.floor(N2 / 2)
    L1_xi1 = np.arange(N1) - np.floor(N1 / 2)
    L2_xi2 = np.arange(N2) - np.floor(N2 / 2)
    xi1 = L1_xi1 / L1
    xi2 = L2_xi2 / L2
    
    g = np.zeros((N1, N2), complex)
    for x1 in range(N1):
        for x2 in range(N2):
            g[x1, x2] = f(L1 * (x1/N1 - 0.5), L2 * (x2/N2 - 0.5))
            g[x1, x2] *= np.exp(u1 * x1/N1 + u2 * x2/N2)
            
    Ff1 = []
    for a in g.T:
        Ff1.append(np.power(-1, L1_xi1) / L1_prime * np.fft.fft(a))
    Ff1 = np.array(Ff1)
    
    Ff = []
    for a in Ff1.T:
        Ff.append(np.power(-1, L2_xi2) / L2_prime * np.fft.fft(a))
    Ff = np.array(Ff).T
    
    return ((xi1, xi2), Ff)


if __name__ == '__main__':
    def f(x1, x2):
        r = np.sqrt(x1**2 + x2**2)
        if x1 <= 0.995 * r or r < 1 or r > 1.3:
            return 0.0
        return max(0,min(2*(r-1),1)) * max(0,min(2*(1.3-r),1)) / r**2
    
    
    ws, Fs = fourier_trans2(f, (50, 50), (50, 50))
    
    ws1, ws2 = np.meshgrid(ws[0], ws[1])
    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection="3d")
    surf = ax.plot_surface(
        ws1, ws2, abs(Fs),
        linewidth=0, antialiased=False, shade=True, alpha=0.5
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.show()
