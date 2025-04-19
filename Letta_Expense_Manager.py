import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from letta_client import Letta
import smtplib
import os
import numpy as np
from datetime import datetime, timedelta
import io

# ---- STREAMLIT PAGE CONFIG ----
st.set_page_config(page_title="Letta Expense Manager", layout="wide")
st.markdown("""
    <style>
    .main {
        background: #f7f9fb;
        padding: 2rem;
        border-radius: 10px;
        max-width: 1200px;
        margin: auto;
        font-family: 'Segoe UI', sans-serif;
    }
    h1, h2, h3 { color: #0A66C2; }
    .stButton>button {
        background-color: #0A66C2;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #084a8f;
    }
    </style>
""", unsafe_allow_html=True)

# ---- SIDEBAR CONFIGURATION ----
st.sidebar.title("⚙️ Configuration")

# Letta Configuration
st.sidebar.header("Letta API")
letta_url = st.sidebar.text_input("Letta API URL", "http://localhost:8283")
letta_agent_id = st.sidebar.text_input("Letta Agent ID", "agent-da571da1-34ae-4d74-8531-046835d8c88d")

# Email Configuration (only shown when needed)
show_email_config = st.sidebar.checkbox("Configure Email")
smtp_server = "smtp.gmail.com"
smtp_port = 587
sender_email = ""
sender_password = ""

if show_email_config:
    st.sidebar.header("Email Settings")
    smtp_server = st.sidebar.text_input("SMTP Server", "smtp.gmail.com")
    smtp_port = st.sidebar.number_input("SMTP Port", value=587)
    sender_email = st.sidebar.text_input("Sender Email")
    sender_password = st.sidebar.text_input("Password", type="password")

# Category Configuration
st.sidebar.header("Categories")
default_categories = {
    "Food": ["restaurant", "cafe", "dining", "meal", "mcdonald", "starbuck"],
    "Travel": ["uber", "lyft", "taxi", "flight", "airbnb", "train", "hotel"],
    "Business": ["software", "subscription", "license", "zoom", "office"],
    "Shopping": ["amazon", "store", "retail", "mall", "target", "walmart"],
    "Groceries": ["grocery", "supermarket", "costco", "safeway", "kroger"],
    "Entertainment": ["netflix", "spotify", "movie", "theater", "music"],
    "Other": []
}

use_custom_categories = st.sidebar.checkbox("Use Custom Categories")
CATEGORY_KEYWORDS = default_categories.copy()

if use_custom_categories:
    st.sidebar.text("Enter comma-separated keywords for each category:")
    for category in default_categories.keys():
        if category != "Other":
            keywords_str = st.sidebar.text_input(
                f"{category} keywords", 
                ",".join(default_categories[category])
            )
            CATEGORY_KEYWORDS[category] = [kw.strip() for kw in keywords_str.split(",") if kw.strip()]

# ---- ADD BUDGET TRACKING FUNCTIONALITY - Place after your Category Configuration section ----
st.sidebar.header("Budget Management")
enable_budgets = st.sidebar.checkbox("Enable Budget Tracking")
budgets = {}

if enable_budgets:
    # Store budgets in session state if not already there
    if "budgets" not in st.session_state:
        st.session_state.budgets = {}
        
    # Budget setting interface
    st.sidebar.subheader("Set Category Budgets")
    for category in CATEGORY_KEYWORDS.keys():
        current_budget = st.session_state.budgets.get(category, 0)
        new_budget = st.sidebar.number_input(
            f"{category} Budget ($)", 
            min_value=0.0, 
            value=float(current_budget),
            step=50.0
        )
        st.session_state.budgets[category] = new_budget
        budgets[category] = new_budget


# ---- ADD SAVINGS GOALS FUNCTIONALITY - Place after Budget Tracking section ----
st.sidebar.header("Savings Goals")
enable_savings = st.sidebar.checkbox("Enable Savings Tracker")
savings_goals = []

if enable_savings:
    # Initialize session state for savings
    if "savings_goals" not in st.session_state:
        st.session_state.savings_goals = []
    
    # Add new goal interface
    st.sidebar.subheader("Add New Goal")
    goal_name = st.sidebar.text_input("Goal Name", key="new_goal_name")
    goal_amount = st.sidebar.number_input("Target Amount ($)", min_value=0.0, key="new_goal_amount")
    goal_date = st.sidebar.date_input("Target Date", key="new_goal_date")
    
    if st.sidebar.button("Add Goal"):
        if goal_name and goal_amount > 0:
            st.session_state.savings_goals.append({
                "name": goal_name,
                "target": goal_amount,
                "date": goal_date,
                "current": 0.0
            })
    
    # Display existing goals
    if st.session_state.savings_goals:
        st.sidebar.subheader("Update Progress")
        for i, goal in enumerate(st.session_state.savings_goals):
            st.sidebar.text(f"{goal['name']} (${goal['current']:.2f} / ${goal['target']:.2f})")
            contribution = st.sidebar.number_input(
                f"Add to {goal['name']}",
                min_value=0.0,
                key=f"contribute_{i}"
            )
            if st.sidebar.button(f"Update {goal['name']}", key=f"update_{i}"):
                st.session_state.savings_goals[i]["current"] += contribution
    
    savings_goals = st.session_state.savings_goals


# ---- ADD THESE FUNCTIONS BEFORE YOUR MAIN APP EXECUTION ----

def detect_recurring_expenses(df):
    """Identify potential recurring expenses in the transaction data"""
    # Group by description and check for regular intervals
    recurring = []
    
    # Convert to proper datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(df["Date"]):
        df["Date"] = pd.to_datetime(df["Date"])
    
    # Group by description
    for desc, group in df.groupby("Description"):
        if len(group) >= 2:  # Need at least 2 occurrences to check pattern
            # Sort by date
            group = group.sort_values("Date")
            
            # Check for consistent amounts
            amounts_consistent = group["Amount"].std() < 1.0  # Small variance in amounts
            
            # Check for consistent time intervals
            dates = group["Date"].dt.to_pydatetime()
            intervals = [(dates[i+1] - dates[i]).days for i in range(len(dates)-1)]
            
            # Check if intervals are consistent (within 3 days tolerance)
            if len(intervals) >= 1:
                interval_consistency = np.std(intervals) <= 3 if len(intervals) > 1 else True
                
                # If both amount and interval are consistent
                if amounts_consistent and interval_consistency:
                    avg_interval = np.mean(intervals) if intervals else 30  # Default to monthly if only one occurrence
                    avg_amount = group["Amount"].mean()
                    
                    # Determine frequency
                    if 25 <= avg_interval <= 35:
                        frequency = "Monthly"
                    elif 6 <= avg_interval <= 8:
                        frequency = "Weekly"
                    elif 13 <= avg_interval <= 16:
                        frequency = "Bi-weekly"
                    elif 85 <= avg_interval <= 95:
                        frequency = "Quarterly"
                    elif 355 <= avg_interval <= 375:
                        frequency = "Yearly"
                    else:
                        frequency = f"Every {int(avg_interval)} days"
                    
                    recurring.append({
                        "Description": desc,
                        "Frequency": frequency,
                        "Amount": avg_amount,
                        "Category": group["Category"].iloc[0],
                        "Last Date": group["Date"].max().strftime("%Y-%m-%d"),
                        "Next Expected": (group["Date"].max() + pd.Timedelta(days=avg_interval)).strftime("%Y-%m-%d")
                    })
    
    return pd.DataFrame(recurring) if recurring else pd.DataFrame()


def forecast_expenses(df, months_ahead=3):
    """Forecast expenses for the next few months based on historical data and recurring expenses"""
    # Identify recurring expenses
    recurring_df = detect_recurring_expenses(df)
    
    # Get the date range
    start_date = df["Date"].max()
    end_date = start_date + pd.DateOffset(months=months_ahead)
    
    # Create forecast dataframe
    forecast_dates = pd.date_range(start=start_date, end=end_date, freq='MS')  # Monthly start
    forecast = pd.DataFrame(forecast_dates, columns=["Date"])
    
    # Initialize with zeros for each category
    for category in df["Category"].unique():
        forecast[category] = 0.0
    
    # For each recurring expense, add to forecast
    for _, rec in recurring_df.iterrows():
        # Parse frequency and calculate occurrences
        freq = rec["Frequency"]
        amount = rec["Amount"]
        category = rec["Category"]
        last_date = pd.to_datetime(rec["Last Date"])
        
        # Calculate estimated occurrences in forecast period
        if freq == "Monthly":
            # Add monthly expense to each month
            for date in forecast_dates:
                forecast.loc[forecast["Date"] == date, category] += amount
        elif freq == "Quarterly":
            # Add quarterly expense
            months_since_last = (forecast_dates[0].to_period('M') - last_date.to_period('M')).n
            for i, date in enumerate(forecast_dates):
                if (i + months_since_last) % 3 == 0:
                    forecast.loc[forecast["Date"] == date, category] += amount
        elif freq == "Yearly":
            # Add yearly expense if it falls within forecast period
            next_occurrence = last_date + pd.DateOffset(years=1)
            if next_occurrence in forecast_dates:
                forecast.loc[forecast["Date"] == next_occurrence, category] += amount
                
    # Add column for total monthly expenses
    forecast["Total"] = forecast.drop("Date", axis=1).sum(axis=1)
    
    return forecast


def detect_anomalies(df, threshold=2.0):
    """Detect unusual spending patterns or potential duplicate payments"""
    anomalies = []
    
    # Check for unusually large transactions by category
    for category in df["Category"].unique():
        cat_df = df[df["Category"] == category]
        if len(cat_df) >= 5:  # Need enough data for meaningful stats
            mean = cat_df["Amount"].mean()
            std = cat_df["Amount"].std()
            upper_limit = mean + (threshold * std)
            
            # Find outliers
            outliers = cat_df[cat_df["Amount"] > upper_limit]
            for _, row in outliers.iterrows():
                anomalies.append({
                    "Type": "Large Transaction",
                    "Description": row["Description"],
                    "Amount": row["Amount"],
                    "Date": row["Date"].strftime("%Y-%m-%d"),
                    "Category": category,
                    "Details": f"${row['Amount']:.2f} is {((row['Amount'] - mean) / std):.1f} standard deviations above the mean (${mean:.2f}) for {category}"
                })
    
    # Check for potential duplicate transactions (same description, amount within 3 days)
    df = df.sort_values("Date")
    for i in range(len(df) - 1):
        current = df.iloc[i]
        for j in range(i+1, min(i+10, len(df))):
            next_row = df.iloc[j]
            days_diff = (next_row["Date"] - current["Date"]).days
            
            # Check if potential duplicate (same description, similar amount, close dates)
            if (days_diff <= 3 and 
                current["Description"] == next_row["Description"] and
                abs(current["Amount"] - next_row["Amount"]) < 1.0):
                
                anomalies.append({
                    "Type": "Potential Duplicate",
                    "Description": current["Description"],
                    "Amount": current["Amount"],
                    "Date": current["Date"].strftime("%Y-%m-%d"),
                    "Category": current["Category"],
                    "Details": f"Similar transaction of ${next_row['Amount']:.2f} on {next_row['Date'].strftime('%Y-%m-%d')} (within {days_diff} days)"
                })
    
    return pd.DataFrame(anomalies) if anomalies else pd.DataFrame()


def calculate_financial_health(df, budgets=None):
    """Calculate a financial health score based on spending patterns"""
    scores = {}
    total_score = 0
    max_score = 0
    
    # 1. Budget adherence (30 points)
    if budgets:
        category_spending = df.groupby("Category")["Amount"].sum()
        budget_scores = []
        
        for category, budget in budgets.items():
            if budget > 0:  # Only consider categories with budgets
                spent = category_spending.get(category, 0)
                ratio = spent / budget
                
                # Score based on how close to budget
                if ratio <= 0.8:  # Under budget by 20%+ (excellent)
                    cat_score = 10
                elif ratio <= 1.0:  # Under budget (good)
                    cat_score = 8
                elif ratio <= 1.1:  # Slightly over budget (okay)
                    cat_score = 5
                elif ratio <= 1.25:  # Over budget (concerning)
                    cat_score = 3
                else:  # Well over budget (poor)
                    cat_score = 0
                    
                budget_scores.append(cat_score)
        
        if budget_scores:
            avg_budget_score = sum(budget_scores) / len(budget_scores)
            budget_total = avg_budget_score * 3  # Scale to 30 points max
            scores["Budget Adherence"] = budget_total
            total_score += budget_total
            max_score += 30
    
    # 2. Expense Distribution (20 points)
    category_distrib = df.groupby("Category")["Amount"].sum() / df["Amount"].sum()
    
    # Check for balanced spending
    if "Food" in category_distrib and category_distrib["Food"] <= 0.15:
        scores["Food Spending"] = 5
        total_score += 5
    else:
        scores["Food Spending"] = 0
        
    if "Entertainment" in category_distrib and category_distrib["Entertainment"] <= 0.1:
        scores["Entertainment Spending"] = 5
        total_score += 5
    else:
        scores["Entertainment Spending"] = 0
        
    if "Shopping" in category_distrib and category_distrib["Shopping"] <= 0.15:
        scores["Shopping Habits"] = 5
        total_score += 5
    else:
        scores["Shopping Habits"] = 0
    
    if "Travel" in category_distrib and category_distrib["Travel"] <= 0.1:
        scores["Travel Expenses"] = 5
        total_score += 5
    else:
        scores["Travel Expenses"] = 0
    
    max_score += 20
    
    # 3. Add points for diversification (20 points)
    category_count = len(df["Category"].unique())
    if category_count >= 6:
        diversification_score = 20
    else:
        diversification_score = category_count * 3
    
    scores["Diversification"] = diversification_score
    total_score += diversification_score
    max_score += 20
    
    # Calculate final score as percentage
    final_score = (total_score / max_score * 100) if max_score > 0 else 0
    
    # Return score and breakdown
    return round(final_score), scores


def generate_financial_tips(df, health_score=None):
    """Generate personalized financial tips based on spending patterns"""
    tips = []
    
    # Get spending by category
    if df.empty:
        return ["Upload transaction data to receive personalized financial tips."]
    
    category_spending = df.groupby("Category")["Amount"].sum()
    
    # Look for patterns and generate tips
    if "Food" in category_spending:
        food_percent = category_spending["Food"] / category_spending.sum() * 100
        if food_percent > 15:
            tips.append("🍽️ Your food spending is higher than recommended (15%). Consider meal planning and cooking at home more often to reduce expenses.")
    
    if "Entertainment" in category_spending:
        entertainment_percent = category_spending["Entertainment"] / category_spending.sum() * 100
        if entertainment_percent > 10:
            tips.append("🎬 Entertainment expenses are on the high side. Look for free or low-cost alternatives for entertainment, such as community events or streaming services instead of multiple subscriptions.")
    
    # Check for savings behavior
    if "Savings" in category_spending:
        savings_percent = category_spending["Savings"] / category_spending.sum() * 100
        if savings_percent < 10:
            tips.append("💰 Your savings rate appears to be below the recommended 10%. Try to automate savings by setting up a recurring transfer to your savings account on payday.")
    else:
        tips.append("💸 No savings category detected. Consider allocating at least 10-20% of your income to savings and investments.")
    
    # Check for recurring subscriptions
    recurring_df = detect_recurring_expenses(df)
    if not recurring_df.empty:
        subscription_count = len(recurring_df[recurring_df["Amount"] < 30])  # Small recurring payments
        if subscription_count > 5:
            tips.append(f"📱 You have {subscription_count} small recurring subscriptions. Consider reviewing these to identify any you don't use regularly and could cancel.")
    
    # Add generic tips based on financial health score
    if health_score is not None:
        if health_score < 50:
            tips.append("⚠️ Your financial health score indicates room for improvement. Focus on creating a budget and tracking expenses closely.")
        elif health_score < 70:
            tips.append("📊 Your financial health is moderate. Consider setting specific financial goals and working toward increasing your savings rate.")
        else:
            tips.append("🌟 Your financial health score is good! Continue your disciplined financial habits and consider increasing investments for long-term wealth building.")
    
    # Add generic tips if not enough personalized ones
    if len(tips) < 3:
        tips.append("💳 Pay off high-interest debt first, especially credit cards, before focusing on other financial goals.")
        tips.append("📈 Consider setting up automatic contributions to retirement accounts to build long-term wealth.")
        tips.append("🛒 Use the 24-hour rule for non-essential purchases: wait 24 hours before buying to reduce impulse spending.")
    
    return tips


def show_budget_progress(filtered_df, budgets):
    st.subheader("Budget Tracking")
    
    if not budgets:
        st.info("No budgets set. Enable budget tracking in the sidebar to monitor your spending against budget targets.")
        return
    
    # Calculate spending by category
    category_spending = filtered_df.groupby("Category")["Amount"].sum()
    
    # Create progress bars for each category with budget
    for category, budget in budgets.items():
        if budget > 0:  # Only show categories with budgets set
            spent = category_spending.get(category, 0)
            percentage = min(100, int((spent / budget) * 100))
            
            # Show warning colors when approaching or exceeding budget
            color = "normal"
            if percentage >= 90:
                color = "error"
            elif percentage >= 75:
                color = "warning"
                
            col1, col2 = st.columns([3, 1])
            with col1:
                st.progress(percentage/100, text=f"{category}: ${spent:.2f} / ${budget:.2f}")
            with col2:
                remaining = budget - spent
                if remaining >= 0:
                    st.write(f"${remaining:.2f} left")
                else:
                    st.write(f"${abs(remaining):.2f} over", unsafe_allow_html=True)


def show_savings_progress(goals):
    st.subheader("Savings Goals Progress")
    
    if not goals:
        st.info("No savings goals set. You can add goals in the sidebar.")
        return
    
    for goal in goals:
        percentage = min(100, int((goal["current"] / goal["target"]) * 100))
        days_left = (goal["date"] - datetime.now().date()).days
        
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            st.progress(percentage/100, text=f"{goal['name']}: ${goal['current']:.2f} / ${goal['target']:.2f}")
        with col2:
            st.write(f"{percentage}% complete")
        with col3:
            if days_left > 0:
                st.write(f"{days_left} days left")
            else:
                st.write("Past due", unsafe_allow_html=True)
                
        # Calculate required saving rate
        if days_left > 0:
            remaining = goal["target"] - goal["current"]
            if remaining > 0:
                daily_rate = remaining / days_left
                monthly_rate = daily_rate * 30
                st.info(f"To reach your goal, you need to save ${daily_rate:.2f} daily or ${monthly_rate:.2f} monthly.")


def add_export_options(df):
    st.subheader("Export Options")
    export_format = st.selectbox(
        "Select export format",
        ["CSV", "Excel", "JSON"]
    )
    
    if st.button(f"Export as {export_format}"):
        if export_format == "CSV":
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download CSV",
                csv,
                "expenses_export.csv",
                "text/csv"
            )
        elif export_format == "Excel":
            # Requires openpyxl package
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name="Expenses")
                # Add summary sheet
                summary = pd.DataFrame({
                    "Category": df.groupby("Category")["Amount"].sum().index,
                    "Total Amount": df.groupby("Category")["Amount"].sum().values
                })
                summary.to_excel(writer, index=False, sheet_name="Summary")
            
            st.download_button(
                "Download Excel",
                buffer.getvalue(),
                "expenses_export.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        elif export_format == "JSON":
            json_str = df.to_json(orient="records", date_format="iso")
            st.download_button(
                "Download JSON",
                json_str,
                "expenses_export.json",
                "application/json"
            )


def add_advanced_filters(df):
    st.subheader("Advanced Filters")
    
    with st.expander("Show Advanced Filters"):
        col1, col2 = st.columns(2)
        
        with col1:
            # Amount range filter
            min_amount = float(df["Amount"].min())
            max_amount = float(df["Amount"].max())
            amount_range = st.slider(
                "Amount Range ($)",
                min_value=min_amount,
                max_value=max_amount,
                value=(min_amount, max_amount),
                step=5.0
            )
            
            # Text search
            search_text = st.text_input("Search in descriptions")
        
        with col2:
            # Day of week filter
            if "Date" in df.columns:
                days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                selected_days = st.multiselect(
                    "Days of week",
                    options=days,
                    default=days
                )
    
    # Apply filters to dataframe
    filtered_df = df.copy()
    
    # Apply amount filter
    filtered_df = filtered_df[(filtered_df["Amount"] >= amount_range[0]) & 
                             (filtered_df["Amount"] <= amount_range[1])]
    
    # Apply text search
    if search_text:
        filtered_df = filtered_df[filtered_df["Description"].str.contains(search_text, case=False)]
    
    # Apply day of week filter
    if "Date" in df.columns and selected_days:
        day_mapping = {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday", 
                      4: "Friday", 5: "Saturday", 6: "Sunday"}
        day_numbers = [list(day_mapping.keys())[list(day_mapping.values()).index(day)] for day in selected_days]
        filtered_df = filtered_df[filtered_df["Date"].dt.weekday.isin(day_numbers)]
    
    return filtered_df


# ---- FUNCTIONS ----
def categorize_expense(description):
    description = str(description).lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword.lower() in description for keyword in keywords):
            return category
    return "Other"

def analyze_trends(df):
    if "Date" not in df.columns:
        return pd.DataFrame()
    
    # Copy to avoid SettingWithCopyWarning
    analysis_df = df.copy()
    analysis_df["Week"] = analysis_df["Date"].dt.strftime("%Y-%U")
    weekly_summary = analysis_df.groupby(["Week", "Category"])["Amount"].sum().unstack().fillna(0)
    return weekly_summary

def plot_spending(df):
    if df.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No data to display", ha='center', va='center')
        return fig
        
    fig, ax = plt.subplots(figsize=(12, 6))
    df.plot(kind="bar", stacked=True, ax=ax, colormap='viridis')
    plt.title("Weekly Spending by Category")
    plt.ylabel("Amount ($)")
    plt.xlabel("Week")
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

def plot_category_pie(df):
    if df.empty or "Category" not in df.columns:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.text(0.5, 0.5, "No category data to display", ha='center', va='center')
        return fig
        
    category_totals = df.groupby("Category")["Amount"].sum()
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(
        category_totals, 
        labels=category_totals.index, 
        autopct='%1.1f%%',
        startangle=90,
        shadow=True
    )
    ax.axis('equal')
    plt.title("Spending by Category")
    return fig

def letta_insights(prompt):
    try:
        # Initialize client with configurable URL
        client = Letta(base_url=letta_url)
        
        response = client.agents.messages.create(
            agent_id=letta_agent_id,
            messages=[{"role": "user", "content": prompt}]
        )
        formatted_response = "\n**Letta Insights:**\n"
        for message in response.messages:
            role = getattr(message, "role", "expense manager").capitalize()
            content = getattr(message, "content", None)
            if isinstance(content, list):
                for item in content:
                    if hasattr(item, "text"):
                        formatted_response += f"\n**{role}:** {item.text.strip()}\n"
            elif isinstance(content, str):
                formatted_response += f"\n**{role}:** {content.strip()}\n"
        return formatted_response.strip()
    except Exception as e:
        return f"Letta failed to generate insights: {e}"

def letta_insights_with_fallback(prompt, df):
    """Try to get insights from Letta, but fall back to basic analysis if unavailable"""
    try:
        # First try regular Letta insights
        letta_response = letta_insights(prompt)
        if "failed" in letta_response:
            raise Exception("Letta API unavailable")
        return letta_response
    except Exception as e:
        st.warning(f"Using fallback analysis instead of Letta: {e}")
        # Fallback to basic statistical analysis
        total_spent = df["Amount"].sum()
        avg_transaction = df["Amount"].mean()
        biggest_expense = df.loc[df["Amount"].idxmax()]
        categories = df.groupby("Category")["Amount"].sum().sort_values(ascending=False)
        
        fallback_analysis = f"""
        **Basic Expense Analysis:**
        
        Total spent: ${total_spent:.2f}
        Average transaction: ${avg_transaction:.2f}
        Largest expense: ${biggest_expense['Amount']:.2f} for {biggest_expense['Description']}
        
        **Spending by Category:**
        """
        
        for category, amount in categories.items():
            percent = (amount / total_spent) * 100
            fallback_analysis += f"\n- {category}: ${amount:.2f} ({percent:.1f}%)"
            
        return fallback_analysis

def send_email_summary(subject, body, to_email):
    try:
        if not sender_email or not sender_password:
            st.warning("Please configure email settings in the sidebar first")
            return False
            
        message = f"Subject: {subject}\n\n{body}"
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, to_email, message)
        return True
    except Exception as e:
        st.error(f"Email failed: {e}")
        return False

def generate_sample_data():
    """Generate sample transaction data for demo purposes"""
    import random
    from datetime import datetime, timedelta
    
    categories = list(CATEGORY_KEYWORDS.keys())
    descriptions = {
        "Food": ["Restaurant Dinner", "Starbucks Coffee", "McDonald's Lunch", "Cafe Brunch"],
        "Travel": ["Uber Ride", "Airbnb Stay", "Taxi Service", "Train Ticket"],
        "Business": ["Zoom Subscription", "Office Software", "Business Lunch", "AWS Services"],
        "Shopping": ["Amazon Purchase", "Target Shopping", "Retail Store", "Mall Shopping"],
        "Groceries": ["Walmart Groceries", "Costco Shopping", "Supermarket Purchase", "Kroger"],
        "Entertainment": ["Netflix Subscription", "Spotify Premium", "Movie Tickets", "Concert"],
        "Other": ["Miscellaneous", "General Purchase", "Service Fee", "Subscription"]
    }
    
    data = []
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    current_date = start_date
    while current_date <= end_date:
        # Add 1-5 transactions per day
        for _ in range(random.randint(1, 5)):
            category = random.choice(categories)
            description = random.choice(descriptions.get(category, descriptions["Other"]))
            amount = round(random.uniform(5, 200), 2)
            
            data.append({
                "Date": current_date.strftime("%Y-%m-%d"),
                "Description": description,
                "Amount": amount,
                "Category": category
            })
        current_date += timedelta(days=1)
    
    return pd.DataFrame(data)

# ---- MAIN APP ----
st.title("📊 Letta-Powered Expense Manager")
st.markdown("Upload your transaction data to get AI-powered insights and expense management.")

# File upload options
upload_option = st.radio(
    "Choose data source:",
    ["Upload CSV file", "Use sample data"],
    horizontal=True
)

df = None

if upload_option == "Upload CSV file":
    uploaded_file = st.file_uploader("Upload your transaction log (CSV)", type=["csv"])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("File uploaded successfully!")
        except Exception as e:
            st.error(f"Error reading file: {e}")
else:
    if st.button("Generate Sample Data"):
        df = generate_sample_data()
        st.success("Sample data generated successfully!")

# Process data if available
if df is not None:
    # Standardize column names
    df.columns = [col.strip().title() for col in df.columns]
    
    # Check for required columns
    required_cols = ["Date", "Description", "Amount"]
    if not all(col in df.columns for col in required_cols):
        st.error(f"CSV must contain these columns: {', '.join(required_cols)}")
    else:
        # Data preprocessing
        try:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df = df.dropna(subset=["Date", "Amount"])
            
            # Apply categorization if not already present
            if "Category" not in df.columns:
                df["Category"] = df["Description"].apply(categorize_expense)
            
            # Display data
            st.subheader("📋 Transaction Data")
            
            # Create tabs for different views
            tab1, tab2, tab3, tab4 = st.tabs(["Data Table", "Spending Analysis", "AI Insights", "Financial Insights"])

            with tab1:
                # Basic filters first
                st.subheader("Basic Filters")
                col1, col2 = st.columns(2)
                
                with col1:
                    # Your existing date filter code
                    min_date = df["Date"].min().date()
                    max_date = df["Date"].max().date()
                    date_range = st.date_input(
                        "Date range",
                        [min_date, max_date],
                        min_value=min_date,
                        max_value=max_date
                    )
                
                with col2:
                    # Your existing category filter code
                    selected_categories = st.multiselect(
                        "Select categories",
                        options=df["Category"].unique(),
                        default=df["Category"].unique()
                    )
                
                # Apply basic filters
                filtered_df = df.copy()
                if len(date_range) == 2:
                    start_date, end_date = date_range
                    filtered_df = filtered_df[
                        (filtered_df["Date"].dt.date >= start_date) & 
                        (filtered_df["Date"].dt.date <= end_date)
                    ]
                
                if selected_categories:
                    filtered_df = filtered_df[filtered_df["Category"].isin(selected_categories)]
                
                # Add advanced filters
                filtered_df = add_advanced_filters(filtered_df)
                
                # Show filtered data
                st.subheader("Transaction Data")
                st.dataframe(filtered_df, use_container_width=True)
                
                # Add export options
                st.markdown("---")
                add_export_options(filtered_df)


            with tab2:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Weekly Spending Trends")
                    trends = analyze_trends(filtered_df)
                    st.pyplot(plot_spending(trends))
                
                with col2:
                    st.subheader("Spending by Category")
                    st.pyplot(plot_category_pie(filtered_df))
                
                # Summary statistics
                st.subheader("Expense Summary")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Expenses", f"${filtered_df['Amount'].sum():.2f}")
                
                with col2:
                    st.metric("Average Transaction", f"${filtered_df['Amount'].mean():.2f}")
                
                with col3:
                    st.metric("Number of Transactions", str(len(filtered_df)))
                
                # Show budget progress if enabled
                if enable_budgets:
                    st.markdown("---")
                    show_budget_progress(filtered_df, budgets)
            
            with tab3:
                st.subheader("🤖 Letta AI Insights")
                
                if st.button("Generate Insights"):
                    with st.spinner("Generating insights with Letta..."):
                        # Prepare data for Letta
                        insight_prompt = "Please provide financial analysis and money-saving tips based on these transactions:\n\n"
                        
                        # Add summary stats
                        insight_prompt += f"Total Spending: ${filtered_df['Amount'].sum():.2f}\n"
                        insight_prompt += "Top Categories:\n"
                        
                        for category, amount in filtered_df.groupby("Category")["Amount"].sum().sort_values(ascending=False).items():
                            insight_prompt += f"- {category}: ${amount:.2f}\n"
                        
                        # Add transaction details (limit to avoid overwhelming)
                        insight_prompt += "\nRecent Transactions:\n"
                        recent_transactions = filtered_df.sort_values("Date", ascending=False).head(20)
                        for _, row in recent_transactions.iterrows():
                            insight_prompt += f"- {row['Date'].strftime('%Y-%m-%d')}: {row['Description']} - ${row['Amount']:.2f} ({row['Category']})\n"
                        
                        # Get insights with fallback option
                        insights = letta_insights_with_fallback(insight_prompt, filtered_df)
                        st.markdown(insights)
                        
                        # Store for email
                        if "insights" not in st.session_state:
                            st.session_state.insights = insights
                
                # Email option
                if "insights" in st.session_state:
                    st.subheader("📧 Email Insights")
                    recipient_email = st.text_input("Recipient Email")
                    
                    if st.button("Send Insights via Email") and recipient_email:
                        # Prepare email content
                        email_subject = f"Expense Analysis - {datetime.now().strftime('%Y-%m-%d')}"
                        email_body = f"Expense Analysis Report\n\n{st.session_state.insights}"
                        
                        # Send email
                        if send_email_summary(email_subject, email_body, recipient_email):
                            st.success("Email sent successfully!")

            with tab4:
                st.subheader("🔄 Recurring Expenses")
                if st.button("Detect Recurring Expenses"):
                    with st.spinner("Analyzing transaction patterns..."):
                        recurring_expenses = detect_recurring_expenses(filtered_df)
                        
                        if not recurring_expenses.empty:
                            st.write("We've identified these potential recurring expenses:")
                            st.dataframe(recurring_expenses)
                            
                            # Calculate total recurring expenses
                            total_recurring = recurring_expenses["Amount"].sum()
                            st.info(f"Total recurring expenses: ${total_recurring:.2f} per month (approx.)")
                        else:
                            st.info("No clear recurring expenses detected in your transaction data.")
                
                st.markdown("---")
                st.subheader("📅 Expense Forecast")
                forecast_months = st.slider("Forecast months ahead", 1, 12, 3)
                
                if st.button("Generate Forecast"):
                    with st.spinner("Generating forecast..."):
                        forecast_df = forecast_expenses(filtered_df, months_ahead=forecast_months)
                        
                        if not forecast_df.empty:
                            st.write("Estimated monthly expenses for the next few months:")
                            st.dataframe(forecast_df)
                            
                            # Visualize the forecast
                            fig, ax = plt.subplots(figsize=(10, 6))
                            forecast_df.set_index("Date").drop("Total", axis=1).plot(
                                kind="bar", stacked=True, ax=ax, colormap='viridis'
                            )
                            plt.title("Forecasted Monthly Expenses")
                            plt.ylabel("Amount ($)")
                            plt.xticks(rotation=45)
                            plt.tight_layout()
                            st.pyplot(fig)
                
                st.markdown("---")
                st.subheader("⚠️ Unusual Spending Patterns")
                if st.button("Detect Anomalies"):
                    with st.spinner("Analyzing transaction patterns..."):
                        anomalies_df = detect_anomalies(filtered_df)
                        
                        if not anomalies_df.empty:
                            st.warning("We've detected some unusual transactions:")
                            st.dataframe(anomalies_df)
                        else:
                            st.success("No unusual spending patterns detected in your transaction data.")
                
                # Show savings goals if enabled
                if enable_savings:
                    st.markdown("---")
                    show_savings_progress(savings_goals)
                
                st.markdown("---")
                st.subheader("💹 Financial Health")
                
                if st.button("Calculate Financial Health Score"):
                    with st.spinner("Analyzing your financial health..."):
                        budgets_data = budgets if enable_budgets else None
                        health_score, score_breakdown = calculate_financial_health(filtered_df, budgets_data)
                        
                        # Create columns for the score and details
                        col1, col2 = st.columns([1, 2])
                        
                        with col1:
                            # Display the score
                            st.markdown(f"""
                            <div style="text-align: center; background-color: {'green' if health_score >= 70 else 'orange' if health_score >= 50 else 'red'}; 
                                        color: white; padding: 20px; border-radius: 10px;">
                                <h1 style="margin: 0;">{health_score}</h1>
                                <p>Financial Health Score</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            # Display the score breakdown
                            st.subheader("Score Breakdown")
                            for category, score in score_breakdown.items():
                                st.metric(category, f"{score:.1f}")
                
                st.markdown("---")
                st.subheader("💡 Financial Tips")
                
                if st.button("Get Financial Tips"):
                    with st.spinner("Generating personalized tips..."):
                        health_score_val = health_score if 'health_score' in locals() else None
                        tips = generate_financial_tips(filtered_df, health_score_val)
                        
                        for tip in tips:
                            st.markdown(f"- {tip}")
                        
                        # Option to get AI-powered tips from Letta
                        if st.button("Get AI-Powered Advice from Letta"):
                            letta_prompt = f"""
                            Based on my financial data:
                            - Total spending: ${filtered_df['Amount'].sum():.2f}
                            - Top spending categories: {', '.join(filtered_df.groupby('Category')['Amount'].sum().nlargest(3).index.tolist())}
                            - Financial health score: {health_score_val if health_score_val else 'Unknown'}
                            
                            Please provide personalized financial advice to improve my financial situation.
                            """
                            
                            letta_advice = letta_insights_with_fallback(letta_prompt, filtered_df)
                            st.markdown(letta_advice)
        
        except Exception as e:
            st.error(f"Error processing data: {e}")
            st.error("Please make sure your data is in the correct format")

# App footer
st.markdown("---")
st.markdown("### 🤖 Powered by Letta AI")
st.markdown("This expense manager uses Letta to provide intelligent insights and automate financial analysis.")