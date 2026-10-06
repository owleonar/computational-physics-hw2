from math import exp

tol = 1.0e-6

h = 6.62607015e-34 # J * s
c = 2.99792458e8 # m / s
k_B = 1.380649e-23 # J / K
lambda_Sun = 502.0e-9  # m

# nonlinear eq. from Planck's law
def f(x):
    return 5.0 * exp(-x) + x - 5.0

# binary search
def search(a, b):
    # make sure first interval isolates a root
    if f(a) * f(b) > 0.0:
        raise ValueError("Interval [a,b] doesn't isolate a root")
    
    # keep bisecting the interval until the midpoint is within our tolerance for the root
    while 0.5 * (b-a) > tol:

        # midpoint rule
        midpoint = 0.5 * (a + b)

        # choose next interval
        if f(a) * f(midpoint) <= 0.0:
            b = midpoint
        else:
            a = midpoint

    # return final midpoint as best approximation of the root
    return 0.5 * (a + b)

def main():

    # x = 0 is a trivial root, so exclude it from first interval
    x = search(0.5,15.0)

    b = (h * c)/(k_B * x)
    T_Sun = b / lambda_Sun

    print(f"RESULT \nx = {x:.6f}\nWien's law displacement constant: b = {b:.6e} m K\nSolar temperature at surface: T = {T_Sun:.1f} K")

if __name__ == "__main__":
    main()