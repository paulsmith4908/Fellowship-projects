import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy.optimize import minimize

# Try importing yfinance, handle if missing
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

# --- CONFIGURATION ---
st.set_page_config(page_title="Junior Quant Lab | Smith Academics", layout="wide")

# --- INITIALIZE SESSION STATE ---
if 'raw_df' not in st.session_state:
    st.session_state.raw_df = None

@st.cache_data
def fetch_yahoo_data(tickers, period="2y"):
    """Fetches closing prices from Yahoo Finance."""
    if not tickers:
        return None
    try:
        data = yf.download(tickers, period=period, group_by='ticker', auto_adjust=True)
        # Handle multi-index columns if multiple tickers
        if len(tickers) > 1:
            df = data.xs('Close', level=1, axis=1)
        else:
            df = data['Close'].to_frame(name=tickers[0])
        
        df.reset_index(inplace=True)
        return df
    except Exception as e:
        return None

# --- MATHEMATICAL ENGINE ---

@st.cache_data
def process_data(df):
    """
    Takes a clean DataFrame (Date index, Numeric columns), calculates stats.
    """
    try:
        # 4. Calculate Returns
        returns = df.pct_change().dropna()
        
        if returns.empty:
            return None, None, None, None, None

        # 5. Annualized Stats (252 trading days)
        means = returns.mean() * 252
        cov_matrix = returns.cov() * 252
        corr_matrix = returns.corr()
        vols = np.sqrt(np.diag(cov_matrix))
        
        return means, vols, cov_matrix, corr_matrix, returns
    except Exception as e:
        return None, None, None, None, None

def get_tangency_portfolio(means, cov_matrix, rf_rate, min_w, max_w):
    """
    Calculates the weights of the Tangency Portfolio (Max Sharpe) using Optimization with Bounds.
    """
    n = len(means)
    
    def negative_sharpe(weights):
        p_ret = np.dot(weights, means)
        p_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return -(p_ret - rf_rate) / p_vol

    # Constraints: Sum of weights = 1
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    # Bounds per asset
    bounds = tuple((min_w, max_w) for _ in range(n))
    
    # Initial Guess (Equal weights)
    init_guess = np.ones(n) / n
    
    result = minimize(negative_sharpe, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    
    if result.success:
        weights = result.x
        port_ret = np.dot(weights, means)
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        port_sharpe = (port_ret - rf_rate) / port_vol
        return weights, port_ret, port_vol, port_sharpe
    else:
        return None, 0, 0, 0

def simulate_monte_carlo(means, cov_matrix, num_simulations, risk_free_rate, min_w, max_w):
    n_assets = len(means)
    
    # REJECTION SAMPLING FOR CONSTRAINED WEIGHTS
    # 1. Generate random weights summing to 1 (using Normal distribution logic)
    # 2. Filter rows that violate min_w/max_w
    # Note: This is computationally inefficient for very tight bounds but fine for visual demos.
    
    batch_size = num_simulations * 10 # Generate extra to account for rejection
    
    if min_w >= 0:
        # If Long Only, Dirichlet is efficient and effective
        alpha = 0.5
        weights = np.random.dirichlet(np.ones(n_assets) * alpha, size=batch_size)
    else:
        # If Shorting allowed, use Gaussian normalization
        # Standard Normal -> Normalize to sum to 1
        raw = np.random.normal(0, 1, (batch_size, n_assets))
        weights = raw / raw.sum(axis=1, keepdims=True)
    
    # Filter bounds
    mask = np.all((weights >= min_w) & (weights <= max_w), axis=1)
    valid_weights = weights[mask]
    
    # If we didn't get enough, just take what we have (or slice)
    if len(valid_weights) > num_simulations:
        valid_weights = valid_weights[:num_simulations]
    elif len(valid_weights) == 0:
         # Fallback if constraints are too tight to find random solutions easily
         # Return equal weights just to not crash
         valid_weights = np.ones((1, n_assets)) / n_assets
    
    # Calculate stats
    port_returns = np.dot(valid_weights, means)
    port_vols = np.sqrt(np.einsum('ij,ji->i', np.dot(valid_weights, cov_matrix), valid_weights.T))
    sharpe_ratios = (port_returns - risk_free_rate) / port_vols
    
    return port_vols, port_returns, sharpe_ratios, valid_weights

def calculate_theoretical_frontier(means, cov_matrix, num_points=200, min_w=0.0, max_w=1.0):
    """
    Calculates the Efficient Frontier using Scipy Optimize with Bounds.
    """
    n = len(means)
    
    # 1. Find Min and Max returns possible in this Constrained world
    # We run two quick optimizations to find the ceiling and floor of returns
    bounds = tuple((min_w, max_w) for _ in range(n))
    constraints_base = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    
    # Max Return Optimization
    res_max = minimize(lambda x: -np.dot(x, means), np.ones(n)/n, method='SLSQP', bounds=bounds, constraints=constraints_base)
    max_ret = -res_max.fun if res_max.success else max(means)
    
    # Min Return (approximate as Min Asset or run optimization if shorting allowed)
    res_min = minimize(lambda x: np.dot(x, means), np.ones(n)/n, method='SLSQP', bounds=bounds, constraints=constraints_base)
    min_ret = res_min.fun if res_min.success else min(means)

    # Pad slightly
    target_returns = np.linspace(min_ret, max_ret, num_points)
    frontier_vols = []
    valid_returns = []
    
    init_guess = np.ones(n) / n
    
    for r in target_returns:
        # Minimize Variance for target return r
        def portfolio_vol(weights):
            return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            
        constraints = (
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},         # Sum weights = 1
            {'type': 'eq', 'fun': lambda x: np.dot(x, means) - r}   # Target Return
        )
        
        result = minimize(portfolio_vol, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
        
        if result.success:
            frontier_vols.append(result.fun)
            valid_returns.append(r)
            
    return frontier_vols, valid_returns

def simulate_vasicek(r0, kappa, theta, sigma, T=1.0, dt=1/252, n_sims=100):
    n_steps = int(T / dt)
    rates = np.zeros((n_steps, n_sims))
    rates[0] = r0
    
    for t in range(1, n_steps):
        rt = rates[t-1]
        drift = kappa * (theta - rt) * dt
        shock = sigma * np.sqrt(dt) * np.random.normal(0, 1, n_sims)
        rates[t] = rt + drift + shock
        
    return rates

# --- UI LAYOUT ---

st.sidebar.title("⚙️ Controls")

# DATA INPUT (Moved to Sidebar)
st.sidebar.subheader("1. Data Source")
data_source = st.sidebar.radio("Input Method:", ["Yahoo Finance", "Upload CSV"], index=0)

tickers_to_process = None

# LOGIC: Load into Session State, Read from Session State
if data_source == "Yahoo Finance":
    if YFINANCE_AVAILABLE:
        # Updated Default Tickers
        default_tickers = "AAPL MSFT GOOG SPY GLD NVDA JNJ"
        ticker_input = st.sidebar.text_area("Enter Tickers (space separated):", default_tickers)
        period = st.sidebar.selectbox("History:", ["1y", "2y", "5y", "10y"], index=1)
        
        # --- AUTO-FETCH LOGIC ---
        # If no data is currently loaded, automatically fetch the defaults on startup.
        if st.session_state.raw_df is None:
            tickers_list = list(set(ticker_input.upper().split()))
            if len(tickers_list) >= 2:
                with st.spinner("Initializing: Fetching market data..."):
                    df = fetch_yahoo_data(tickers_list, period)
                    if df is not None:
                        st.session_state.raw_df = df

        if st.sidebar.button("Fetch Data"):
            tickers_list = list(set(ticker_input.upper().split()))
            if len(tickers_list) < 2:
                st.sidebar.error("Please enter at least 2 tickers.")
            else:
                with st.spinner("Fetching data from Yahoo Finance..."):
                    df = fetch_yahoo_data(tickers_list, period)
                    if df is not None:
                        st.session_state.raw_df = df
    else:
        st.sidebar.error("`yfinance` library not installed. Please run `pip install yfinance`.")

elif data_source == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload Price CSV", type=["csv"])
    if uploaded_file:
        try:
            # Only reload if the file object actually changed to prevent re-reads
            df = pd.read_csv(uploaded_file, na_values=['#N/A', 'nan'])
            st.session_state.raw_df = df
        except Exception as e:
            st.error(f"Read error: {e}")

# Use the data from session state
if st.session_state.raw_df is not None:
    raw_df = st.session_state.raw_df.copy()
else:
    raw_df = None

# --- MAIN PAGE ---

st.title("📊 Junior Quant Lab")
st.markdown("""
**Portfolio Construction & Risk Engine.** *By Paul Smith | Smith Academics*
""")

tab1, tab2 = st.tabs(["Efficient Frontier", "Rates (Vasicek)"])

with tab1:
    # Process Data if available
    processed_successfully = False
    means, vols, cov_matrix, corr_matrix, returns_df = None, None, None, None, None
    date_col_name = "Date" # Default name for display
    
    if raw_df is not None:
        # Clean Data
        # Locate the date column safely
        date_col = next((col for col in raw_df.columns if 'date' in col.lower()), raw_df.columns[0])
        date_col_name = date_col # Store name for display later
        
        raw_df[date_col] = pd.to_datetime(raw_df[date_col], errors='coerce')
        raw_df.set_index(date_col, inplace=True)
        raw_df = raw_df.apply(pd.to_numeric, errors='coerce').dropna(axis=1, how='all').ffill()

        real_means, real_vols, real_cov, real_corr, real_returns = process_data(raw_df)
        
        if real_means is not None:
            # Filtering Logic
            real_tickers = real_returns.columns.tolist()
            
            # If coming from Yahoo/Demo, we usually want all of them unless user filters
            st.subheader("Asset Universe")
            selected_tickers = st.multiselect(
                "Active Assets:", 
                real_tickers, 
                default=real_tickers
            )
            
            if len(selected_tickers) >= 2:
                indices = [real_tickers.index(t) for t in selected_tickers]
                means = real_means.iloc[indices].values
                vols = real_vols[indices]
                cov_matrix = real_cov.iloc[indices, indices].values
                corr_matrix = real_corr.iloc[indices, indices]
                returns_df = real_returns[selected_tickers]
                processed_successfully = True

    # Main Visuals
    if processed_successfully:
        # ROW 1: Chart and Metrics
        col_main, col_metrics = st.columns([2.8, 1.2])

        # Sidebar params specific to Tab 1
        st.sidebar.divider()
        st.sidebar.subheader("2. Optimization Params")
        rf_rate = st.sidebar.slider("Risk-Free Rate ($R_f$)", 0.0, 0.10, 0.045, 0.005)
        n_sims = st.sidebar.slider("Simulations", 500, 5000, 2000, 500)
        show_theory = st.sidebar.checkbox("Show Efficient Frontier", True)
        
        st.sidebar.markdown("**Position Limits (Constraints)**")
        min_w = st.sidebar.number_input("Min Weight (-0.5 = -50%)", value=-0.5, step=0.1)
        max_w = st.sidebar.number_input("Max Weight (1.5 = 150%)", value=1.5, step=0.1)

        # Variables for best portfolio
        display_ret = 0.0
        display_vol = 0.0
        display_sharpe = 0.0
        display_weights = None

        with col_main:
            mc_vols, mc_rets, mc_sharpes, mc_weights = simulate_monte_carlo(means, cov_matrix, n_sims, rf_rate, min_w, max_w)
            
            # --- PLOTLY CHART ---
            fig = go.Figure()

            # 1. Monte Carlo Cloud
            fig.add_trace(go.Scatter(
                x=mc_vols, y=mc_rets,
                mode='markers',
                marker=dict(size=5, color=mc_sharpes, colorscale='Spectral_r', showscale=True, colorbar=dict(title="Sharpe")),
                name='Feasible Portfolios',
                hovertemplate='Risk: %{x:.1%}<br>Return: %{y:.1%}<br>Sharpe: %{marker.color:.2f}<extra></extra>'
            ))

            # 2. Efficient Frontier (Theoretical)
            if show_theory:
                # INCREASED POINTS TO 200 FOR SMOOTHER CURVE
                th_vols, th_rets = calculate_theoretical_frontier(means, cov_matrix, num_points=200, min_w=min_w, max_w=max_w)
                
                # Calculate Exact Tangency Portfolio
                tan_weights, tan_ret, tan_vol, tan_sharpe = get_tangency_portfolio(means, cov_matrix, rf_rate, min_w, max_w)
                
                if len(th_vols) > 0:
                    fig.add_trace(go.Scatter(
                        x=th_vols, y=th_rets,
                        mode='lines', line=dict(color='white', width=3), 
                        name='Efficient Frontier'
                    ))
                    
                    if tan_weights is not None:
                        # Use the EXACT Tangency numbers
                        display_ret = tan_ret
                        display_vol = tan_vol
                        display_sharpe = tan_sharpe
                        display_weights = tan_weights

                        fig.add_trace(go.Scatter(
                            x=[tan_vol], y=[tan_ret],
                            mode='markers', marker=dict(size=14, symbol='star', color='yellow', line=dict(width=1, color='black')),
                            name='Max Sharpe Portfolio'
                        ))
                        
                        # CML
                        fig.add_trace(go.Scatter(
                            x=[0, tan_vol * 1.5], y=[rf_rate, rf_rate + (tan_ret - rf_rate)/tan_vol * (tan_vol * 1.5)],
                            mode='lines', line=dict(color='yellow', width=1, dash='dash'),
                            name='CML'
                        ))
            else:
                # Fallback to Best Simulation if theory is off
                if len(mc_sharpes) > 0:
                    best_sim_idx = np.argmax(mc_sharpes)
                    display_ret = mc_rets[best_sim_idx]
                    display_vol = mc_vols[best_sim_idx]
                    display_sharpe = mc_sharpes[best_sim_idx]
                    display_weights = mc_weights[best_sim_idx]

            # 3. Individual Assets
            fig.add_trace(go.Scatter(
                x=vols, y=means,
                mode='markers+text', text=selected_tickers, textposition="top center",
                marker=dict(size=10, color="white", line=dict(width=1, color='black')),
                name='Assets'
            ))

            fig.update_layout(
                title="Mean-Variance Landscape",
                xaxis=dict(title="Annualized Risk (σ)", tickformat=".1%"),
                yaxis=dict(title="Annualized Return (μ)", tickformat=".1%"),
                height=600,
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor='rgba(0,0,0,0.5)'),
                margin=dict(l=0, r=0, t=50, b=0)
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_metrics:
            st.markdown("### Optimal Portfolio")
            st.caption("Tangency (Max Sharpe)")
            
            c1, c2 = st.columns(2)
            c1.metric("Max Sharpe", f"{display_sharpe:.2f}")
            c2.metric("Return", f"{display_ret:.1%}")
            st.metric("Volatility", f"{display_vol:.1%}")
            
            if display_weights is not None:
                st.divider()
                st.markdown("**Allocation**")
                
                # Create a DataFrame for the weights
                weights_df = pd.DataFrame({
                    'Asset': selected_tickers,
                    'Weight': display_weights
                }).sort_values(by='Weight', ascending=True) # Ascending for correct visual order in horizontal bar
                
                # Horizontal Bar Chart
                fig_weights = px.bar(
                    weights_df, 
                    x='Weight', 
                    y='Asset',
                    text_auto='.1%',
                    color='Weight',
                    color_continuous_scale='Viridis',
                    orientation='h' # <--- HORIZONTAL
                )
                fig_weights.update_layout(
                    showlegend=False, 
                    height=250, 
                    margin=dict(l=0, r=0, t=0, b=0),
                    xaxis=dict(showgrid=False, tickformat=".0%"),
                    coloraxis_showscale=False
                )
                fig_weights.update_traces(textposition='inside')
                st.plotly_chart(fig_weights, use_container_width=True)
                
                if np.any(display_weights < 0):
                    st.caption("⚠️ Short Selling Enabled")

            st.divider()
            st.markdown("**Tail Risk (95% Daily)**")
            
            if display_weights is not None:
                port_daily_rets = returns_df.dot(display_weights)
            else:
                eq_weights = np.ones(len(selected_tickers)) / len(selected_tickers)
                port_daily_rets = returns_df.dot(eq_weights)
            
            var_95 = np.percentile(port_daily_rets, 5)
            cvar_95 = port_daily_rets[port_daily_rets <= var_95].mean()
            
            c3, c4 = st.columns(2)
            c3.metric("VaR", f"{var_95:.2%}")
            c4.metric("CVaR", f"{cvar_95:.2%}")

        # ROW 2: Correlation and Data (Side by Side)
        st.divider()
        col_bottom_left, col_bottom_right = st.columns([1, 1])

        with col_bottom_left:
            st.markdown("##### 🔗 Correlation Matrix")
            fig_corr = go.Figure(data=go.Heatmap(
                z=corr_matrix.values, x=selected_tickers, y=selected_tickers,
                colorscale='RdBu', zmin=-1, zmax=1,
                text=np.round(corr_matrix.values, 2), texttemplate="%{text}"
            ))
            fig_corr.update_layout(height=400, margin=dict(t=0, b=0, l=0, r=0))
            st.plotly_chart(fig_corr, use_container_width=True)

        with col_bottom_right:
            st.markdown("##### 📋 Market Data Inspector")
            tab_prices, tab_rets = st.tabs(["Price History", "Daily Returns"])
            
            with tab_prices:
                st.dataframe(
                    raw_df.reset_index(), 
                    use_container_width=True,
                    height=350,
                    column_config={
                        date_col_name: st.column_config.DateColumn("Date", format="YYYY-MM-DD")
                    }
                )
            
            with tab_rets:
                st.dataframe(
                    returns_df.head(100).reset_index(), 
                    use_container_width=True,
                    height=350,
                    column_config={
                        date_col_name: st.column_config.DateColumn("Date", format="YYYY-MM-DD")
                    }
                )

    else:
        st.info("👈 Please load data from the Sidebar to begin.")

# --- TAB 2: STOCHASTIC RATES ---
with tab2:
    col_v_controls, col_v_graph = st.columns([1, 3])
    
    with col_v_controls:
        st.header("Vasicek Parameters")
        r0 = st.number_input("Current Rate ($r_0$)", 0.0, 0.15, 0.05, 0.005, format="%.3f")
        theta = st.number_input("Long-Term Mean ($\\theta$)", 0.0, 0.15, 0.04, 0.005, format="%.3f")
        kappa = st.slider("Reversion Speed ($\kappa$)", 0.1, 5.0, 1.5, 0.1)
        sigma_r = st.slider("Rate Volatility ($\sigma$)", 0.0, 0.10, 0.02, 0.005, format="%.3f")
        sim_years = st.slider("Years", 1, 10, 3)
        
    with col_v_graph:
        paths = simulate_vasicek(r0, kappa, theta, sigma_r, T=sim_years, n_sims=100)
        x_axis = np.linspace(0, sim_years, len(paths))
        
        fig_v = go.Figure()
        for i in range(min(50, paths.shape[1])):
            fig_v.add_trace(go.Scatter(x=x_axis, y=paths[:, i], mode='lines', line=dict(color='rgba(0,255,200,0.1)'), showlegend=False, hoverinfo='skip'))
            
        fig_v.add_trace(go.Scatter(x=x_axis, y=np.mean(paths, axis=1), mode='lines', line=dict(color='white', width=3), name='Mean Path'))
        fig_v.add_trace(go.Scatter(x=[0, sim_years], y=[theta, theta], mode='lines', line=dict(color='yellow', dash='dash'), name='Long-term Mean'))
        
        fig_v.update_layout(title="Vasicek Interest Rate Simulation", xaxis_title="Years", yaxis_title="Rate", height=500)
        st.plotly_chart(fig_v, use_container_width=True)