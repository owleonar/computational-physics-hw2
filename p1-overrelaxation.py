from math import exp

c = 2.0
tol = 1.0e-6

def f(x):
    return 1.0 - exp(-c * x)

def df(x):
    return c * exp(-c * x)

# one relaxation function for handling both ordinary and overrelaxation methods
# x0 is initial guess for x' = f(x)
# omega is overrelaxation parameter (omega = 0 is ordinary relaxation)
def relaxation(omega, x0 = 0.9):
    x = x0

    for iteration in range(1,10000):

        # overrelaxation iteration equation
        x_new = (1.0 + omega) * f(x) - omega * x

        # derivative of overrelaxation iteration equation w.r.t. x
        slope = (1.0 + omega) * df(x) - omega

        # error estimate from part (a)
        error = abs( (x - x_new) * slope / (slope - 1.0) )

        if error < tol:
            return x_new, iteration
        
        x = x_new

def main():

    # ordinary relaxation
    x, iterations = relaxation(0.0)

    print(f"Ordinary relaxation: \nx = {x:.6f} \niterations = {iterations}")

    # overrelaxation
    print("Overrelaxation:")

    for omega in (0.1, 0.4, 0.5, 0.6, 0.7, 1.0):

        x, iterations = relaxation(omega)

        print(f"omega = {omega:0.1f}: x = {x:0.6f}, iterations = {iterations}")

if __name__ == "__main__":
    main()