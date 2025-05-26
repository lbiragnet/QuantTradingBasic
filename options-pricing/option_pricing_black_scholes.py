#########################################
#                                       #
#  BLACK-SCHOLES MODEL OPTION PRICING   #
#                                       #
#########################################


# ---------------------------- IMPORTS ---------------------------- #

import numpy as np
from scipy.stats import norm
from typing import Tuple


# ---------------------------- OPTION PRICING ---------------------------- #


class BlackScholesOption:
    """
    Pricing for an option under the Black-Scholes model. Handles cases where the underlying
    has a continuous dividend yield and cases where there is no dividend yield.
    """

    def __init__(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: None | float = 0,
    ):
        """
        Initialise BlackScholesOption object.
        Args:
            | S (float): Underlying spot price.
            | K (float): Option strike price.
            | T (float): Option time to expiration.
            | r (float): Risk-free rate.
            | sigma (float): Volatility.
            | q (float): Dividend yield.
        """
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.q = q if q is not None else 0.0

    def __str__(self):
        """
        Print details of the BlackScholesOption object.
        """
        return {
            "Underlying price: ": self.S,
            "Strike price: ": self.K,
            "Time to maturity: ": self.T,
            "Risk-free rate: ": self.r,
            "Volatility: ": self.sigma,
            "Dividend yield: ": self.q,
        }

    def calc_d1(self) -> float:
        """
        Calculate d1 value for option under Black-Scholes model.
        Returns:
            | (float): Value for d1 in the Black-Scholes formula.
        """
        try:
            return (
                np.log(self.S / self.K)
                + (self.r - self.q + (self.sigma**2) / 2) * self.T
            ) / (self.sigma * np.sqrt(self.T))
        except Exception as e:
            print(f"Error when calculating d1 value: {e}.")

    def calc_d2(self, d1) -> float:
        """
        Calculate d2 value for option under Black-Scholes model.
        Args:
            | d1 (float): Value for d1 in the Black-Scholes formula.
        Returns:
            | (float): Value for d2 in the Black-Scholes formula.
        """
        try:
            return d1 - self.sigma * np.sqrt(self.T)
        except Exception as e:
            print(f"Error when calculating d2 value: {e}.")

    def price_call_option(self, d1, d2) -> float:
        """
        Calculate call price for option under Black-Scholes model.
        Args:
            | d1 (float): Value for d1 in the Black-Scholes formula.
            | d2 (float): Value for d2 in the Black-Scholes formula.
        Returns:
            | (float): Call option price.
        """
        try:
            return (np.exp(-self.q * self.T) * self.S * norm.cdf(d1)) - (
                np.exp(-self.r * self.T) * self.K * norm.cdf(d2)
            )
        except Exception as e:
            print(f"Error when calculating call price for option: {e}.")

    def price_put_option(self, d1, d2) -> float:
        """
        Calculate put price for option under Black-Scholes model.
        Args:
            | d1 (float): Value for d1 in the Black-Scholes formula.
            | d2 (float): Value for d2 in the Black-Scholes formula.
        Returns:
            | (float): Put option price.
        """
        try:
            return (np.exp(-self.r * self.T) * self.K * norm.cdf(-d2)) - (
                np.exp(-self.q * self.T) * self.S * norm.cdf(-d1)
            )
        except Exception as e:
            print(f"Error when calculating put price for option: {e}.")

    def calc_delta(self, d1) -> Tuple[float, float]:
        """
        Calculate delta - first derivative of option price with respect to underlying price.
        Returns a tuple of delta values for call and put options.
        Args:
            | d1 (float): Value for d1 in the Black-Scholes formula.
        Returns:
            | (Tuple[float, float]): Tuple of call delta and put delta.
        """
        try:
            delta_call = np.exp(-self.q * self.T) * norm.cdf(d1)
            delta_put = np.exp(-self.q * self.T) * (norm.cdf(d1) - 1)
            return (delta_call, delta_put)
        except Exception as e:
            print(f"Error when calculating delta for option: {e}.")

    def calc_gamma(self, d1) -> float:
        """
        Calculate gamma - second derivative of option price with respect to underlying price.
        Same value for calls and puts.
        Args:
            | d1 (float): Value for d1 in the Black-Scholes formula.
        Returns:
            | (float): Call and put gamma.
        """
        try:
            phi = (1 / np.sqrt(2 * np.pi)) * np.exp(-(d1**2) / 2)
            return (np.exp(-self.q * self.T) * phi) / (
                self.S * self.sigma * np.sqrt(self.T)
            )
        except Exception as e:
            print(f"Error when calculating gamma for option: {e}.")

    def calc_theta(self, d1, d2, days) -> Tuple[float, float]:
        """
        Calculate theta - first derivative of option price with respect to time to expiration.
        Args:
            | d1 (float): Value for d1 in the Black-Scholes formula.
            | d2 (float): Value for d2 in the Black-Scholes formula.
            | days (int): Days used for calculation (in full year / trade year).
        Returns:
            | (Tuple[float, float]): Tuple of call theta and put theta.
        """
        try:
            phi = (1 / np.sqrt(2 * np.pi)) * np.exp(-(d1**2) / 2)
            theta_call = (1 / days) * (
                -(
                    (self.S * self.sigma * np.exp(-self.q * self.T) * phi)
                    / (2 * np.sqrt(self.T))
                    - (self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(d2))
                    + (self.q * self.S * np.exp(-self.q * self.T) * norm.cdf(d1))
                )
            )
            theta_put = (1 / days) * (
                -(
                    (self.S * self.sigma * np.exp(-self.q * self.T) * phi)
                    / (2 * np.sqrt(self.T))
                    + (self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-d2))
                    - (self.q * self.S * np.exp(-self.q * self.T) * norm.cdf(-d1))
                )
            )
            return (theta_call, theta_put)
        except Exception as e:
            print(f"Error when calculating theta for option: {e}.")

    def calc_vega(self, d1) -> Tuple[float, float]:
        """
        Calculate vega - first derivative of option price with respect to volatility.
        Same value for calls and puts (given as a percentage).
        Args:
            | d1 (float): Value for d1 in the Black-Scholes formula.
        Returns:
            | (float): Call and put vega.
        """
        phi = (1 / np.sqrt(2 * np.pi)) * np.exp(-(d1**2) / 2)
        return (1 / 100) * self.S * np.exp(-self.q * self.T) * np.sqrt(self.T) * phi

    def calc_rho(self, d2):
        """
        Calculate rho - first derivative of option price with respect to interest rate.
        Given as a percentage.
        Args:
            | d2 (float): Value for d2 in the Black-Scholes formula.
        Returns:
            | (Tuple[float, float]): Tuple of call rho and put rho.
        """
        rho_call = (1 / 100) * self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(d2)
        rho_put = (1 / 100) * self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-d2)
        return (rho_call, rho_put)
