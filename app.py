from flask import Flask, render_template, request
import numpy as np
from scipy.stats import norm
import yfinance as yf

app = Flask(__name__)


def black_scholes_call(S, K, T, r, sigma):
    """
    Calculates the price of a European call option using the Black-Scholes model.

    Parameters:
    S (float): Current stock price
    K (float): Strike price of the option
    T (float): Time to expiration (in years)
    r (float): Risk-free interest rate (annualized)
    sigma (float): Volatility of the underlying stock (annualized)

    Returns:
    float: Price of the call option
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price


def geometric_brownian_motion(S, T, r, sigma, n_simulations, n_steps):
    """
    Generates Monte Carlo simulations for geometric Brownian motion.

    Parameters:
    S (float): Current stock price
    T (float): Time to expiration (in years)
    r (float): Risk-free interest rate (annualized)
    sigma (float): Volatility of the underlying stock (annualized)
    n_simulations (int): Number of simulations
    n_steps (int): Number of time steps

    Returns:
    ndarray: Array of simulated stock prices
    """
    dt = T / n_steps
    sim_prices = np.zeros((n_simulations, n_steps + 1))
    sim_prices[:, 0] = S

    for i in range(n_simulations):
        for j in range(1, n_steps + 1):
            z = np.random.standard_normal()
            sim_prices[i, j] = sim_prices[i, j - 1] * np.exp((r - 0.5 * sigma ** 2) * dt +
                                                             sigma * np.sqrt(dt) * z)

    return sim_prices


def european_call_option_price(S, K, T, r, sigma, n_simulations=15000, n_steps=100):
    """
    Estimates the price of a European call option using Monte Carlo simulation.

    Parameters:
    S (float): Current stock price
    K (float): Strike price of the option
    T (float): Time to expiration (in years)
    r (float): Risk-free interest rate (annualized)
    sigma (float): Volatility of the underlying stock (annualized)
    n_simulations (int): Number of simulations
    n_steps (int): Number of time steps

    Returns:
    float: Estimated price of the call option
    """
    sim_prices = geometric_brownian_motion(S, T, r, sigma, n_simulations, n_steps)
    payoffs = np.maximum(sim_prices[:, -1] - K, 0)
    option_price = np.exp(-r * T) * np.mean(payoffs)
    return option_price


def get_stock_info(stock_name, risk_free_rate):
    try:
        # Get stock data
        stock_data = yf.Ticker(stock_name)
        # Get historical data
        hist = stock_data.history(period="5y")
        # Calculate daily returns
        daily_returns = hist['Close'].pct_change().dropna()
        # Calculate volatility (standard deviation of returns)
        volatility = np.std(daily_returns) * np.sqrt(252)  # 252 trading days in a year
        # Get current stock price
        current_price = stock_data.history(period='1d')['Close'].iloc[-1]

        return {
            'volatility': volatility,
            'current_price': current_price,
            'risk_free_rate': risk_free_rate
        }
    except Exception as e:
        print("Error:", e)
        return None


def get_risk_free_rate():
    try:
        # Get the data for the 10-Year Treasury Yield (^TNX)
        treasury_data = yf.Ticker("^TNX")
        # Get the current yield
        current_yield = treasury_data.history(period='1y')['Close'].iloc[-1]

        return current_yield / 100  # Convert percentage to decimal
    except:
        try:
            # Get the data for the 1-Year Treasury Constant Maturity Rate (DGS1)
            treasury_data = yf.Ticker("DGS1")
            # Get the current yield
            current_yield = treasury_data.history(period='1y')['Close'].iloc[-1]

            return current_yield / 100  # Convert percentage to decimal
        except Exception as e:
            print("Error:", e)
            return 0.05


@app.route('/', methods=['GET', 'POST'])
def calculate_option_price():
    if request.method == 'POST':
        option_type = request.form['option_type']
        if option_type == 'manual_black_scholes':
            S = float(request.form['S'])
            K = float(request.form['K'])
            T = float(request.form['T'])
            r = float(request.form['r'])
            sigma = float(request.form['sigma'])

            call_price = black_scholes_call(S, K, T, r, sigma)
            return render_template('result.html', call_price=call_price)

        elif option_type == 'manual_monte_carlo':
            S = float(request.form['S'])
            K = float(request.form['K'])
            T = float(request.form['T'])
            r = float(request.form['r'])
            sigma = float(request.form['sigma'])

            call_price = european_call_option_price(S, K, T, r, sigma)
            return render_template('result.html', call_price=call_price)

        elif option_type == 'stock_name_black_scholes':
            stock_name = request.form['stock_name']
            K = float(request.form['K_stock'])
            T = float(request.form['T_stock'])

            risk_free_rate = get_risk_free_rate()
            print(risk_free_rate)
            stock_info = get_stock_info(stock_name, risk_free_rate)
            if stock_info:
                S = stock_info['current_price']
                r = stock_info['risk_free_rate']
                sigma = stock_info['volatility']

                call_price = black_scholes_call(S, K, T, r, sigma)
                return render_template('result.html', call_price=call_price)

        elif option_type == 'stock_name_monte_carlo':
            stock_name = request.form['stock_name']
            K = float(request.form['K_stock'])
            T = float(request.form['T_stock'])

            risk_free_rate = get_risk_free_rate()
            print(risk_free_rate)

            stock_info = get_stock_info(stock_name, risk_free_rate)
            if stock_info:
                S = stock_info['current_price']
                r = stock_info['risk_free_rate']
                sigma = stock_info['volatility']

                call_price = european_call_option_price(S, K, T, r, sigma)
                return render_template('result.html', call_price=call_price)
            else:
                return "Error retrieving stock information"

    return render_template('index.html')


if __name__ == '__main__':
    # app.run(debug=True)
    app.run(host='0.0.0.0', port=8000)