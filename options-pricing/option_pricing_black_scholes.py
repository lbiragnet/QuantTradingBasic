#########################################
#                                       #
#  BLACK-SCHOLES MODEL OPTION PRICING   #
#                                       #
#########################################


# ---------------------------- IMPORTS ---------------------------- #

import numpy as np
from scipy.stats import norm


# ---------------------------- OPTION CLASS ---------------------------- #


class BlackScholesEuropeanOption:
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
        option_type: str,
        q: None | float = 0,
    ):
        """
        Initialise BlackScholesEuropeanOption object.
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
        self.option_type = option_type
        self.d1 = self.calc_d1()
        self.d2 = self.calc_d2()
        self.price = self.calc_price()
        self.delta = self.calc_delta()
        self.theta = self.calc_theta()
        self.vega = self.calc_vega()
        self.rho = self.calc_rho()
        self.epsilon = self.calc_epsilon()
        self.gamma = self.calc_gamma()
        self.vomma = self.calc_vomma()
        self.vanna = self.calc_vanna()
        self.charm = self.calc_charm()
        self.veta = self.calc_veta()
        self.vera = self.calc_vera()
        self.speed = self.calc_speed()
        self.zomma = self.calc_zomma()
        self.color = self.calc_color()
        self.ultima = self.calc_ultima()

    def __str__(self):
        """
        Print details of the BlackScholesEuropeanOption object.
        """
        return {
            "Underlying price: ": self.S,
            "Strike price: ": self.K,
            "Time to maturity: ": self.T,
            "Risk-free rate: ": self.r,
            "Volatility: ": self.sigma,
            "Dividend yield: ": self.q,
        }

    # ---------------------------- OPTION PRICING ---------------------------- #

    def calc_d1(self) -> float:
        """
        Calculate d1 value for option under Black-Scholes model.
        Returns:
            | (float): Value for d1 in the Black-Scholes formula.
        """
        try:
            d1 = (
                np.log(self.S / self.K)
                + (self.r - self.q + (self.sigma**2) / 2) * self.T
            ) / (self.sigma * np.sqrt(self.T))
            return d1
        except Exception as e:
            print(f"Error when calculating d1 value: {e}.")

    def calc_d2(self) -> float:
        """
        Calculate d2 value for option under Black-Scholes model.
        Returns:
            | (float): Value for d2 in the Black-Scholes formula.
        """
        try:
            d2 = self.d1 - self.sigma * np.sqrt(self.T)
            return d2
        except Exception as e:
            print(f"Error when calculating d2 value: {e}.")

    def calc_price(self) -> float | None:
        """
        Calculate the call or put price using the Black-Scholes formula.
        Returns:
            (float): Option price if successful, None otherwise.
        """
        if self.option_type not in {"Call", "Put"}:
            print(
                f"Error - invalid option type: {self.option_type}. Must be 'Call' or 'Put'."
            )
            return None
        try:
            if self.option_type == "Call":
                price = np.exp(-self.q * self.T) * self.S * norm.cdf(self.d1) - np.exp(
                    -self.r * self.T
                ) * self.K * norm.cdf(self.d2)
            else:
                price = np.exp(-self.r * self.T) * self.K * norm.cdf(-self.d2) - np.exp(
                    -self.q * self.T
                ) * self.S * norm.cdf(-self.d1)
            return price
        except Exception as e:
            print(f"Error when pricing {self.option_type} option: {e}")
            return None

    # ---------------------------- FIRST-ORDER GREEKS ---------------------------- #

    def calc_delta(self) -> float:
        """
        Calculate delta - first derivative of option price with respect to underlying price.
        Different formula for calls and puts.
        Returns:
            | (float): Option delta.
        """
        if self.option_type not in {"Call", "Put"}:
            print(
                f"Error - invalid option type: {self.option_type}. Must be 'Call' or 'Put'."
            )
            return None
        try:
            if self.option_type == "Call":
                delta = np.exp(-self.q * self.T) * norm.cdf(self.d1)
            else:
                delta = np.exp(-self.q * self.T) * (norm.cdf(self.d1) - 1)
            return delta
        except Exception as e:
            print(f"Error when calculating delta for option: {e}.")

    def calc_theta(self) -> float:
        """
        Calculate theta - first derivative of option price with respect to time to expiration.
        Different formula for calls and puts.
        Returns:
            | (float): Option theta.
        """
        if self.option_type not in {"Call", "Put"}:
            print(
                f"Error - invalid option type: {self.option_type}. Must be 'Call' or 'Put'."
            )
            return None
        try:
            phi = (1 / np.sqrt(2 * np.pi)) * np.exp(-(self.d1**2) / 2)
            if self.option_type == "Call":
                theta = (
                    -(self.S * self.sigma * np.exp(-self.q * self.T) * phi)
                    / (2 * np.sqrt(self.T))
                    - (self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(self.d2))
                    + (self.q * self.S * np.exp(-self.q * self.T) * norm.cdf(self.d1))
                )
            else:
                theta = (
                    (self.S * self.sigma * np.exp(-self.q * self.T) * phi)
                    / (2 * np.sqrt(self.T))
                    + (self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-self.d2))
                    - (self.q * self.S * np.exp(-self.q * self.T) * norm.cdf(-self.d1))
                )
            return theta
        except Exception as e:
            print(f"Error when calculating theta for option: {e}.")

    def calc_vega(self) -> float:
        """
        Calculate vega - first derivative of option price with respect to volatility.
        Same value for calls and puts (given as a percentage).
        Returns:
            | (float): Option vega.
        """
        try:
            phi = (1 / np.sqrt(2 * np.pi)) * np.exp(-(self.d1**2) / 2)
            vega = self.S * np.exp(-self.q * self.T) * np.sqrt(self.T) * phi
            return vega
        except Exception as e:
            print(f"Error when calculating vega for option: {e}.")

    def calc_rho(self):
        """
        Calculate rho - first derivative of option price with respect to interest rate.
        Given as a percentage.
        Returns:
            | (float): Option rho if successful. None otherwise.
        """
        if self.option_type not in {"Call", "Put"}:
            print(
                f"Error - invalid option type: {self.option_type}. Must be 'Call' or 'Put'."
            )
            return None
        try:
            if self.option_type == "Call":
                rho = (
                    (1 / 100)
                    * self.K
                    * self.T
                    * np.exp(-self.r * self.T)
                    * norm.cdf(self.d2)
                )
            else:
                rho = (
                    (1 / 100)
                    * self.K
                    * self.T
                    * np.exp(-self.r * self.T)
                    * norm.cdf(-self.d2)
                )
            return rho
        except Exception as e:
            print(f"Error when calculating rho for option: {e}.")

    def calc_epsilon(self):
        """
        Calculate epsilon - first derivative of option price with respect to dividend yield.
        Given as a percentage change.
        Returns:
            | (float): Option epsilon if successful. None otherwise.
        """
        if not self.q:
            return 0.0
        if self.option_type not in {"Call", "Put"}:
            print(
                f"Error - invalid option type: {self.option_type}. Must be 'Call' or 'Put'."
            )
            return None
        try:
            if self.option_type == "Call":
                epsilon = (
                    -self.S * self.T * np.exp(-self.q * self.T) * norm.cdf(self.d1)
                )
            else:
                epsilon = (
                    self.S * self.T * np.exp(-self.q * self.T) * norm.cdf(-self.d1)
                )
            return epsilon
        except Exception as e:
            print(f"Error when calculating epsilon for option: {e}.")

    # ---------------------------- SECOND-ORDER GREEKS ---------------------------- #

    def calc_gamma(self) -> float:
        """
        Calculate gamma - second derivative of option price with respect to underlying price.
        Same value for calls and puts.
        Returns:
            | (float): Option gamma.
        """
        try:
            phi = (1 / np.sqrt(2 * np.pi)) * np.exp(-(self.d1**2) / 2)
            gamma = (np.exp(-self.q * self.T) * phi) / (
                self.S * self.sigma * np.sqrt(self.T)
            )
            return gamma
        except Exception as e:
            print(f"Error when calculating gamma for option: {e}.")

    def calc_vomma(self):
        """
        Calculate vomma/volga - second derivative of option price with respect to volatility.
        Returns:
            | (float): Option vomma/volga if successful. None otherwise.
        """
        try:
            vomma = self.vega / self.sigma * self.d1 * self.d2
            return vomma
        except Exception as e:
            print(f"Error when calculating vomma for option: {e}.")

    def calc_vanna(self):
        """
        Calculate vanna - second derivative of option price with respect to 1) underlying price
        and 2) volatility.
        Returns:
            | (float): Option vanna if successful. None otherwise.
        """
        try:
            vanna = self.vega * self.d2 / (self.S * self.sigma * np.sqrt(self.T))
            return vanna
        except Exception as e:
            print(f"Error when calculating vanna for option: {e}.")

    def calc_charm(self):
        """
        Calculate charm - second derivative of option price with respect to 1) underlying price
        and 2) passage of time.
        Returns:
            | (float): Option charm if successful. None otherwise.
        """
        if self.option_type not in {"Call", "Put"}:
            print(
                f"Error - invalid option type: {self.option_type}. Must be 'Call' or 'Put'."
            )
            return None
        try:
            common = (
                np.exp(-self.q * self.T)
                * norm.pdf(self.d1)
                * (
                    2 * (self.r - self.q) * self.T
                    - self.d2 * self.sigma * np.sqrt(self.T)
                )
                / 2
                * self.T
                * self.sigma
                * np.sqrt(self.T)
            )
            if self.option_type == "Call":
                charm = self.q * np.exp(-self.q * self.T) * norm.cdf(self.d1) - common
            else:
                charm = -self.q * np.exp(-self.q * self.T) * norm.cdf(-self.d1) - common
            return charm
        except Exception as e:
            print(f"Error when calculating charm for option: {e}.")

    def calc_veta(self):
        """
        Calculate veta - second derivative of option price with respect to 1) volatility
        and 2) passage of time.
        Returns:
            | (float): Option veta if successful. None otherwise.
        """
        try:
            return (
                -self.S
                * np.exp(-self.q * self.T)
                * norm.pdf(self.d1)
                * np.sqrt(self.T)
                * (
                    self.q
                    + (self.r - self.q) * self.d1 / self.sigma * np.sqrt(self.T)
                    - (1 + self.d1 * self.d2) / 2 * self.T
                )
            )
        except Exception as e:
            print(f"Error when calculating veta for option: {e}.")

    def calc_vera(self):
        """
        Calculate vera - second derivative of option price with respect to 1) volatility
        and 2) interest rate.
        Returns:
            | (float): Option vera if successful. None otherwise.
        """
        try:
            return (
                -self.K
                * self.T
                * np.exp(-self.r * self.T)
                * norm.pdf(self.d2)
                * self.d1
                / self.sigma
            )
        except Exception as e:
            print(f"Error when calculating vera for option: {e}.")

    # ---------------------------- THIRD-ORDER GREEKS ---------------------------- #

    def calc_speed(self):
        """
        Calculate speed - third derivative of option price with respect to the underlying spot price.
        Returns:
            | (float): Option speed if successful. None otherwise.
        """
        try:
            return -self.gamma / self.S * (self.d1 / self.sigma * np.sqrt(self.T) + 1)
        except Exception as e:
            print(f"Error when calculating speed for option: {e}.")

    def calc_zomma(self):
        """
        Calculate zomma - third derivative of option price with respect to  1) the underlying spot price,
        2) the underlying spot price once more and 3) volatility.
        Returns:
            | (float): Option zomma if successful. None otherwise.
        """
        try:
            return self.gamma * (self.d1 * self.d2 - 1) / self.sigma
        except Exception as e:
            print(f"Error when calculating zomma for option: {e}.")

    def calc_color(self):
        """
        Calculate color - third derivative of option price with respect to  1) the underlying spot price,
        2) the underlying spot price once more and 3) passage of time.
        Returns:
            | (float): Option color if successful. None otherwise.
        """
        try:
            phi = lambda x: np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)
            term1 = (
                -np.exp(-self.q * self.T)
                * phi(self.d1)
                / (2 * self.S * self.T * self.sigma * np.sqrt(self.T))
            )
            term2 = (
                2 * self.q * self.T
                + 1
                + (
                    (
                        2 * (self.r - self.q) * self.T
                        - self.d2 * self.sigma * np.sqrt(self.T)
                    )
                    / (self.sigma * np.sqrt(self.T))
                )
                * self.d1
            )
            return term1 * term2
        except Exception as e:
            print(f"Error when calculating color for option: {e}.")

    def calc_ultima(self):
        """
        Calculate ultima - third derivative of option price with respect to volatility
        Returns:
            | (float): Option ultima if successful. None otherwise.
        """
        try:
            return -self.vomma / self.sigma * (1 - self.d1 * self.d2)

        except Exception as e:
            print(f"Error when calculating ultima for option: {e}.")
