import numpy as np
import matplotlib.pyplot as plt

LABEL_SIZE = 16
TITLE_SIZE = 16
TICK_LABEL_SIZE = 13
LEGEND_SIZE = 13

MAJOR_TICK_LENGTH = 6
MINOR_TICK_LENGTH = 3
MAJOR_TICK_WIDTH = 1.0
MINOR_TICK_WIDTH = 0.8

data = np.loadtxt("data/smf_cosmos.dat")

logM = data[:, 0] # dex
n = data[:, 1] # n(Mgal) galaxy stellar mass function, 1 / (dex * Volume)
sigma = data[:, 2] # error in n(Mgal)

h = 1.0e-5
tol = 1.0e-5

# numerical grad from centered difference method (partial derivatives)
# f is a function of len(p) parameters expressed as the parameter vector p
def grad(f, p):
    
    # initialize with all zeros
    grad = np.zeros(len(p))

    # centered difference in each parameter
    for i in range(len(p)):

        # copy p so we don't mess with the original vector
        p_plus = p.copy()
        p_minus = p.copy()

        # shift ith parameter by finite diff step size
        p_plus[i] += h
        p_minus[i] -= h

        # ith component of grad is the finite diff approx in the ith parameter
        grad[i] = (f(p_plus) - f(p_minus))/(2.0 * h)

    return grad

# multidimensional gradient descent (initial parameter vector guess p0)
def grad_descent(f, p0, step=0.1):

    # convert p0 into a float vector
    p = np.array(p0, dtype=float)

    # save value of f and p at each valid p
    vals = [f(p)]
    path = [p.copy()]

    # iterate gradient descent
    for _ in range(10000):

        # compute grad and magnitude of grad
        grad_val = grad(f,p)
        grad_norm = np.linalg.norm(grad_val)

        # stop once within tol for final estimate
        if grad_norm < tol:
            break

        # extract grad direction (direction of steepest increase)
        direction = grad_val / grad_norm

        # initial step size might overshoot the minimum
        # reduce the step size until the trial point leads to function decrease
        step_try = step
        while step_try > 1.0e-8:

            # opposite the grad direction is downhill
            p_new = p - step_try * direction

            # stop once we have decrease
            if f(p_new) < f(p):
                break

            # half step and try again
            step_try *= 0.5
        
        if step_try <= 1.0e-8:
            break

        # update parameter vec to the new accepted one
        p = p_new

        # append value of f and p
        vals.append(f(p))
        path.append(p.copy())

    # return best fit, and also the history of f and p
    return p, np.array(vals), np.array(path)

# given test func
def test_f(p):
    x, y = p
    return (x - 2.0)**2 + (y-2.0)**2

# Schechter mass func in log
def schechter(logM, p):
    log_phi, log_Mstar, alpha = p

    phi = 10.0**log_phi
    ratio = 10.0**(logM-log_Mstar)

    return np.log(10.0) * phi * ratio**(alpha + 1.0) * np.exp(-ratio)

# Schechter chi-sq
def chi_squared(p):

    model = schechter(logM, p)

    return np.sum( ((n - model)/sigma)**2 )

def main():

    # test grad descent on the test function
    minimum, values, path = grad_descent(test_f, [-1.0, 0.0], step=0.2)

    print(f"TEST FUNCTION \nMinimum: (x0,y0) = ({minimum[0]:.4f},{minimum[1]:.4f})\nf(x0,y0) = {values[-1]:.4e}\n")

    x = np.linspace(-1.5, 3.5, 300)
    y = np.linspace(-1.5, 3.5, 300)
    
    X, Y = np.meshgrid(x, y)
    Z = (X - 2.0)**2 + (Y - 2.0)**2

    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    # Function landscape
    contour = ax.contourf(X, Y, Z, levels=20)
    ax.contour(X, Y, Z, levels=20, linewidths=0.7)

    # Gradient-descent trajectory
    ax.plot(
        path[:, 0],
        path[:, 1],
        "o-",
        markersize=5,
        linewidth=1.5,
        label="Gradient-descent path",
    )

    # Mark initial point and known minimum
    ax.plot(
        path[0, 0],
        path[0, 1],
        "s",
        markersize=8,
        label="Initial point",
    )

    ax.plot(
        2.0,
        2.0,
        "*",
        markersize=12,
        label="Minimum",
    )

    ax.set_xlabel(r"$x$", fontsize=LABEL_SIZE)
    ax.set_ylabel(r"$y$", fontsize=LABEL_SIZE)

    ax.minorticks_on()
    ax.tick_params(
        which="major",
        direction="in",
        top=True,
        right=True,
        labelsize=TICK_LABEL_SIZE,
        length=MAJOR_TICK_LENGTH,
        width=MAJOR_TICK_WIDTH,
    )
    ax.tick_params(
        which="minor",
        direction="in",
        top=True,
        right=True,
        length=MINOR_TICK_LENGTH,
        width=MINOR_TICK_WIDTH,
    )

    ax.legend(fontsize=LEGEND_SIZE)

    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label(r"$F(x,y)$", fontsize=LABEL_SIZE)
    cbar.ax.tick_params(
        which="major",
        labelsize=TICK_LABEL_SIZE,
        length=MAJOR_TICK_LENGTH,
        width=MAJOR_TICK_WIDTH,
    )

    fig.tight_layout()
    fig.savefig("results/gradient-descent-test.pdf")
    plt.close(fig)

    # Schechter fit from grad descent, multiple starting pts

    starts = [ [-2.3, 10.9, -1.2], [-2.5, 11.0, -1.0], [-2.8, 11.1, -0.8] ]
    fits = []

    print("SCHECHTER FIT\n")

    for i, start in enumerate(starts, 1):

        p, values, path = grad_descent(chi_squared, start, step=0.1)
        fits.append((p, values))

        phi = 10.0**p[0]
        Mstar = 10.0**p[1]

        print(f"Start {i}\nphi* = {phi:.6e}\nM* = {Mstar:.6e}\nalpha = {p[2]:.6f}\nchi_sq = {values[-1]:.6f}\n")        

    plt.figure(figsize=(6, 4.5))

    for i, (p, values) in enumerate(fits, 1):
        plt.plot(range(len(values)), values, label=f"Start {i}")

    plt.yscale("log")
    plt.xlim(0, 50)

    plt.xlabel(r"Iteration $i$", fontsize=LABEL_SIZE)
    plt.ylabel(r"$\chi^2$", fontsize=LABEL_SIZE)

    plt.minorticks_on()
    plt.tick_params(which="major", direction="in", top=True, right=True,
                    labelsize=TICK_LABEL_SIZE, length=MAJOR_TICK_LENGTH, width=MAJOR_TICK_WIDTH)
    plt.tick_params(which="minor", direction="in", top=True, right=True,
                    length=MINOR_TICK_LENGTH, width=MINOR_TICK_WIDTH)

    plt.legend(frameon=False, fontsize=LEGEND_SIZE)

    plt.tight_layout()
    plt.savefig("results/chi-squared-convergence.pdf")
    plt.close()

    # best-fit Schechter
    best = fits[0][0]

    logM_model = np.linspace(logM.min(), logM.max(), 500)
    M = 10.0**logM
    M_model = 10.0**logM_model

    plt.figure(figsize=(6, 4.5))

    plt.errorbar(
        M,
        n,
        yerr=sigma,
        fmt="o",
        markersize=5,
        capsize=3,
        label="COSMOS data"
    )

    plt.plot(
        M_model,
        schechter(logM_model, best),
        label="Best-fit Schechter function"
    )

    plt.xscale("log")
    plt.yscale("log")

    plt.xlabel(r"$M_{\rm gal}$", fontsize=LABEL_SIZE)
    plt.ylabel(r"$n(M_{\rm gal})$", fontsize=LABEL_SIZE)

    plt.minorticks_on()
    plt.tick_params(which="major", direction="in", top=True, right=True,
                    labelsize=TICK_LABEL_SIZE, length=MAJOR_TICK_LENGTH, width=MAJOR_TICK_WIDTH)
    plt.tick_params(which="minor", direction="in", top=True, right=True,
                    length=MINOR_TICK_LENGTH, width=MINOR_TICK_WIDTH)

    plt.legend(frameon=False, fontsize=LEGEND_SIZE)

    plt.tight_layout()
    plt.savefig("results/schechter-fit.pdf")
    plt.close()


if __name__ == "__main__":
    main()