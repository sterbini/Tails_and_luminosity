# %%
import numpy as np
from scipy.special import gamma
from matplotlib import pyplot as plt
from scipy.special import hyp2f1

def q_log(x, q):
    """Compute the q-logarithm."""
    if q == 1:
        return np.log(x)
    else:
        return (x**(1 - q) - 1) / (1 - q)


def q_exponential(x, q):
    """Computes the q-exponential function."""
    if q==1:
        return np.exp(x)
    else:
        valid_base = 1 + (1 - q) * x
        valid_base[valid_base < 0] = 0
        return valid_base ** (1 / (1 - q))

def C_q(q):
    """Computes the normalization factor C_q."""
    if  q < 1:
        return (2 * np.sqrt(np.pi) * gamma(1 / (1 - q))) / ((3 - q) * np.sqrt(1 - q) * gamma((3 - q) / (2 * (1 - q))))
    elif q == 1:
        return np.sqrt(np.pi)
    elif 1 < q < 3:
        return (np.sqrt(np.pi) * gamma((3 - q) / (2 * (q - 1)))) / (np.sqrt(q - 1) * gamma(1 / (q - 1)))
    else:
        raise ValueError("q must be in the range (0, 3).")

def q_gaussian(x, q, beta):
    """Computes the q-Gaussian probability density function."""
    if beta<= 0:
        raise ValueError('beta should be positive')
    if q>3:
        raise ValueError('q should be <3.')
    if 1<=q<3:
        return (np.sqrt(beta) / C_q(q)) * q_exponential(-beta * x**2, q)
    if q<1:
        my_limit = 1/np.sqrt(beta*(1-q))
        return np.where(
        (-my_limit <= x) & (x <= my_limit), 
        (np.sqrt(beta) / C_q(q)) * q_exponential(-beta * x**2, q),
        0
    )

def Q_x(x, q, beta):
    return beta*(q-1)*x**2+1

def Q_x_x0(x, x0, q, beta):
    return beta*(3*q-5)*x**2+beta*(6-4*q)*x0**2-1

def A_plus(x, x0, q, beta):
    return hyp2f1(1, (q-2)/(q-1), 1/2, (Q_x(x, q, beta)-Q_x(x0, q, beta))/Q_x(x, q, beta))

def A_minus(x, x0, q, beta):
    return hyp2f1(1, (q-2)/(q-1), -1/2, (Q_x(x, q, beta)-Q_x(x0, q, beta))/Q_x(x, q, beta))

def rho_x_x0(x, x0, q, beta):
    num = Q_x(x0, q, beta)**(1/(1-q)+1/2)*(Q_x_x0(x, x0, q, beta)* A_plus(x, x0, q, beta)+Q_x(x0, q, beta)*A_minus(x, x0, q, beta))
    dem = np.pi*np.sqrt(x0**2-x**2)*Q_x(x, q, beta)
    result = -num/dem
    return np.nan_to_num(result, nan=0.0)

def generalized_box_muller(q, beta, size=1):
    """Generate q-Gaussian deviates using the generalized Box-Muller transform."""
    if beta<= 0:
        raise ValueError('beta should be positive')
    q_prime = (1 + q) / (3 - q)  # Compute q'
    
    # Generate uniform random numbers
    U1 = np.random.uniform(0, 1, size)
    U2 = np.random.uniform(0, 1, size)
    # Compute the transformation
    factor = np.sqrt(-2 / (beta*(3-q)) * q_log(U1, q_prime))
    X = factor * np.cos(2 * np.pi * U2)
    Y = factor * np.sin(2 * np.pi * U2)
    
    return X, Y

# %%
# plotting the q-Gaussian distribution
q = 1.2
beta = 2.0
x = np.linspace(-10, 10, 1000)
y = q_gaussian(x, q, beta)

plt.plot(x, y, label=f'q-Gaussian (q={q}, beta={beta})')
plt.title('q-Gaussian Distribution')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.legend()
plt.grid()
plt.show()
# %%
# fit y with a Gaussian distribution
from scipy.optimize import curve_fit    
def gaussian1(x, sigma):
    """Standard Gaussian function."""
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x) / sigma) ** 2)  

def fit_gaussian1(x, y):
    """Fit a Gaussian to the data."""
    # Initial guess for mu and sigma
    sigma_guess = np.std(x)
    
    # Fit the Gaussian
    # A bounded to ensure A is between 0 and 1
    popt, _ = curve_fit(gaussian1, x, y, p0=[sigma_guess])
    
    return popt  # Returns mu and sigma

def gaussian2(x, A, sigma1, sigma2):
    """Standard Gaussian function."""
    return (
        A*(1 / (sigma1 * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x) / sigma1) ** 2)  +
    (1-A)*(1 / (sigma2 * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x) / sigma2) ** 2))

def fit_gaussian2(x, y):
    """Fit a Gaussian to the data."""
    # Initial guess for mu and sigma
    sigma_guess = np.std(x)
    
    # Fit the Gaussian
    # A bounded to ensure A is between 0 and 1
    p0 = [0.5, sigma_guess, sigma_guess]  # Initial guess
    bounds = ([-1, 0, 0], [1, 10, 10])  # Bounds for A, sigma1, sigma2
    popt, _ = curve_fit(gaussian2, x, y, p0=[.3, sigma_guess, sigma_guess], bounds=bounds)
    
    return popt  # Returns mu and sigma

# Example usage of the fit_gaussian function
A_2, sigma1_2, sigma2_2 = fit_gaussian2(x, y)
sigma_1 = fit_gaussian1(x, y)
plt.plot(x, y, label='q-Gaussian')
plt.plot(x, gaussian2(x, A_2, sigma1_2, sigma2_2), label='Fitted Gaussian', linestyle='--')
# %%
# plot difference between q-Gaussian and Gaussian
plt.plot(x, y - gaussian2(x, A_2, sigma1_2, sigma2_2), label='Difference (q-Gaussian - Gaussian)')
plt.plot(x, y - gaussian1(x,sigma_1), label='Difference (q-Gaussian - Gaussian)')

# %%
# q-Guassian are not closed for overlapping: the VdM scan of two q-Gaussians is not a q-Gaussian. But is very similar
x = np.linspace(-20, 20, 10000)
q1 = 1.
beta1 = 0.5
q2 = 1.5
beta2 = 0.5

plt.figure(figsize=(10, 6))
plt.plot(x, q_gaussian(x-1, q1, beta1), label='q-Gaussian')
plt.plot(x, q_gaussian(x+1, q2, beta2), label='q-Gaussian')
my_list = []

my_offset_list = np.linspace(-10, 10, 10000)
for offset in my_offset_list:
    #compute the integral of the product  between the two q-Gaussians
    my_list.append(np.trapz(q_gaussian(x-offset, q1, beta1)*q_gaussian(x+offset, q2, beta2), x))



# %%
plt.plot(my_offset_list,my_list,'.')
# fit my_list with a q-Gaussian
from scipy.optimize import curve_fit
def q_gaussian_fit(x, A, q, beta):
    """Fit a q-Gaussian to the data."""
    return A*q_gaussian(x, q, beta)       

popt, _ = curve_fit(q_gaussian_fit, my_offset_list, my_list, p0=[1, 1.5, 2.0], bounds=([0, .8, 0], [np.inf, 2.9, np.inf]))
A, q, beta = popt
print(f'Fitted A: {A}, q: {q}, beta: {beta}')

plt.plot(my_offset_list, q_gaussian(my_offset_list, q, beta)*A, label='q-Gaussian')

plt.xlim(-5,5)
# %%
plt.plot(my_offset_list, my_list - q_gaussian(my_offset_list, q, beta)*A, label='q-Gaussian')


# %%
# is the q gaussian closed under sum?
# the sum of two q-Gaussians is not a q-Gaussian, but it is very similar
# we can compute the sum of two q-Gaussians with the
def sum_q_gaussians(x, q1, beta1, mu1, q2, beta2, mu2):
    """Compute the sum of two q-Gaussians."""
    return q_gaussian(x-mu1, q1, beta1) + q_gaussian(x-mu2, q2, beta2)

# Example usage
q1 = 1.2
beta1 = 0.5
mu1 = -1
q2 = 1.2
beta2 = 0.5
mu2=  1
x = np.linspace(-10, 10, 1000)
y = sum_q_gaussians(x, q1, beta1, mu1, q2, beta2, mu2)      
plt.plot(x, y, label='Sum of q-Gaussians')
plt.title('Sum of Two q-Gaussians')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.legend()
plt.grid()
plt.show()  

popt, _ = curve_fit(q_gaussian_fit, x, y, p0=[1, 1.5, 2.0], bounds=([0, .8, 0], [np.inf, 2.9, np.inf]))
A, q, beta = popt
print(f'Fitted A: {A}, q: {q}, beta: {beta}')

# %%
from scipy import integrate

def f(w, z, y, x):
    return np.exp(-(x**2 + y**2 + z**2 + w**2))



bounds = [[-2, 2], [-2, 2], [-5, 5], [-5, 5]]
result, error = integrate.nquad(f, bounds)
print(f"Result of the integral: {result}, Estimated error: {error}")
# %%
import vegas
import numpy as np

# Define the integrand for VEGAS (takes a single argument, a 1D array of variables)
def integrand(x):
    x1, y1, z1, t1 = x  # Unpack the 4D input
    return np.exp(x1**2 + 2*y1**2 + 3*z1**2 + 4*t1**2)

# Create the VEGAS integrator with integration limits for each variable
integ = vegas.Integrator([
    [-1, 1],  # x
    [-1, 1],  # y
    [-1, 1],  # z
    [-1, 1]   # t
])

# Warm-up iterations (to optimize the grid)
integ(integrand, nitn=5, neval=1000)

# Real integration passes
result = integ(integrand, nitn=10, neval=10000)

# Display result
print("Integral =", result.mean, "+/-", result.sdev)
# %%
