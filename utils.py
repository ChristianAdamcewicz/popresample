import numpy as np
from scipy.integrate import cumtrapz
from scipy.special import spence as PL

def inverse_transform_sample(pdf, x):
    ''''''
    cdf = cumtrapz(pdf, x, initial=0)
    cdf /= np.max(cdf)
    cdf = np.nan_to_num(cdf)
    cdf_sample = np.random.rand()
    sample = np.interp(cdf_sample, cdf, x)
    return sample

def trapz_exp(log_y, x=None, dx=1.0, axis=-1):
    ''''''
    y = np.exp(log_y)
    y_int = np.trapz(y, x, dx, axis)
    return np.log(y_int)

def Di(z):
    '''
    Used in chi_effective_prior_from_isotropic_spins function.
    '''
    return PL(1.-z+0j)

def chi_effective_prior_from_isotropic_spins(q,aMax,chi_eff):
    """
    Adapted from https://github.com/tcallister/effective-spin-priors/blob/main/priors.py
    Function defining the conditional priors p(chi_eff|q) corresponding to
    uniform, isotropic component spin priors.
    Inputs
    q: Mass ratio value (according to the convention q<1)
    aMax: Maximum allowed dimensionless component spin magnitude
    xs: Chi_effective value or values at which we wish to compute prior
    Returns:
    Array of prior values
    """

    # Ensure that `xs` and 'qs' is an array and take absolute value
    xs = np.reshape(np.abs(chi_eff),-1)
    qs = np.reshape(q,-1)


    # Set up various piecewise cases
    pdfs = np.ones(xs.size,dtype=complex)*(-1.)
    caseZ = (xs==0)
    caseA = (xs>0)*(xs<aMax*(1.-qs)/(1.+qs))*(xs<qs*aMax/(1.+qs))
    caseB = (xs<aMax*(1.-qs)/(1.+qs))*(xs>qs*aMax/(1.+qs))
    caseC = (xs>aMax*(1.-qs)/(1.+qs))*(xs<qs*aMax/(1.+qs))
    caseD = (xs>aMax*(1.-qs)/(1.+qs))*(xs<aMax/(1.+qs))*(xs>=qs*aMax/(1.+qs))
    caseE = (xs>aMax*(1.-qs)/(1.+qs))*(xs>aMax/(1.+qs))*(xs<aMax)
    caseF = (xs>=aMax)

    # Select relevant effective spins and mass ratios
    x_A = xs[caseA]
    x_B = xs[caseB]
    x_C = xs[caseC]
    x_D = xs[caseD]
    x_E = xs[caseE]

    q_Z = qs[caseZ]
    q_A = qs[caseA]
    q_B = qs[caseB]
    q_C = qs[caseC]
    q_D = qs[caseD]
    q_E = qs[caseE]

    pdfs[caseZ] = (1.+q_Z)/(2.*aMax)*(2.-np.log(q_Z))

    pdfs[caseA] = (1.+q_A)/(4.*q_A*aMax**2)*(
                    q_A*aMax*(4.+2.*np.log(aMax) - np.log(q_A**2*aMax**2 - (1.+q_A)**2*x_A**2))
                    - 2.*(1.+q_A)*x_A*np.arctanh((1.+q_A)*x_A/(q_A*aMax))
                    + (1.+q_A)*x_A*(Di(-q_A*aMax/((1.+q_A)*x_A)) - Di(q_A*aMax/((1.+q_A)*x_A)))
                    )

    pdfs[caseB] = (1.+q_B)/(4.*q_B*aMax**2)*(
                    4.*q_B*aMax
                    + 2.*q_B*aMax*np.log(aMax)
                    - 2.*(1.+q_B)*x_B*np.arctanh(q_B*aMax/((1.+q_B)*x_B))
                    - q_B*aMax*np.log((1.+q_B)**2*x_B**2 - q_B**2*aMax**2)
                    + (1.+q_B)*x_B*(Di(-q_B*aMax/((1.+q_B)*x_B)) - Di(q_B*aMax/((1.+q_B)*x_B)))
                    )

    pdfs[caseC] = (1.+q_C)/(4.*q_C*aMax**2)*(
                    2.*(1.+q_C)*(aMax-x_C)
                    - (1.+q_C)*x_C*np.log(aMax)**2.
                    + (aMax + (1.+q_C)*x_C*np.log((1.+q_C)*x_C))*np.log(q_C*aMax/(aMax-(1.+q_C)*x_C))
                    - (1.+q_C)*x_C*np.log(aMax)*(2. + np.log(q_C) - np.log(aMax-(1.+q_C)*x_C))
                    + q_C*aMax*np.log(aMax/(q_C*aMax-(1.+q_C)*x_C))
                    + (1.+q_C)*x_C*np.log((aMax-(1.+q_C)*x_C)*(q_C*aMax-(1.+q_C)*x_C)/q_C)
                    + (1.+q_C)*x_C*(Di(1.-aMax/((1.+q_C)*x_C)) - Di(q_C*aMax/((1.+q_C)*x_C)))
                    )

    pdfs[caseD] = (1.+q_D)/(4.*q_D*aMax**2)*(
                    -x_D*np.log(aMax)**2
                    + 2.*(1.+q_D)*(aMax-x_D)
                    + q_D*aMax*np.log(aMax/((1.+q_D)*x_D-q_D*aMax))
                    + aMax*np.log(q_D*aMax/(aMax-(1.+q_D)*x_D))
                    - x_D*np.log(aMax)*(2.*(1.+q_D) - np.log((1.+q_D)*x_D) - q_D*np.log((1.+q_D)*x_D/aMax))
                    + (1.+q_D)*x_D*np.log((-q_D*aMax+(1.+q_D)*x_D)*(aMax-(1.+q_D)*x_D)/q_D)
                    + (1.+q_D)*x_D*np.log(aMax/((1.+q_D)*x_D))*np.log((aMax-(1.+q_D)*x_D)/q_D)
                    + (1.+q_D)*x_D*(Di(1.-aMax/((1.+q_D)*x_D)) - Di(q_D*aMax/((1.+q_D)*x_D)))
                    )

    pdfs[caseE] = (1.+q_E)/(4.*q_E*aMax**2)*(
                    2.*(1.+q_E)*(aMax-x_E)
                    - (1.+q_E)*x_E*np.log(aMax)**2
                    + np.log(aMax)*(
                        aMax
                        -2.*(1.+q_E)*x_E
                        -(1.+q_E)*x_E*np.log(q_E/((1.+q_E)*x_E-aMax))
                        )
                    - aMax*np.log(((1.+q_E)*x_E-aMax)/q_E)
                    + (1.+q_E)*x_E*np.log(((1.+q_E)*x_E-aMax)*((1.+q_E)*x_E-q_E*aMax)/q_E)
                    + (1.+q_E)*x_E*np.log((1.+q_E)*x_E)*np.log(q_E*aMax/((1.+q_E)*x_E-aMax))
                    - q_E*aMax*np.log(((1.+q_E)*x_E-q_E*aMax)/aMax)
                    + (1.+q_E)*x_E*(Di(1.-aMax/((1.+q_E)*x_E)) - Di(q_E*aMax/((1.+q_E)*x_E)))
                    )

    pdfs[caseF] = 0.

    # Deal with spins on the boundary between cases
    if np.any(pdfs==-1):
        boundary = (pdfs==-1)
        pdfs[boundary] = 0.5*(chi_effective_prior_from_isotropic_spins(q[boundary],aMax,xs[boundary]+1e-6)\
                        + chi_effective_prior_from_isotropic_spins(q[boundary],aMax,xs[boundary]-1e-6))

    return np.reshape(np.real(pdfs), chi_eff.shape)