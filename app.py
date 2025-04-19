# # Letta-Powered Expense Manager (CSV → Insights)
# # Dependencies: Letta API, Pandas, Matplotlib, Streamlit, smtplib

# import pandas as pd
# import streamlit as st
# import matplotlib.pyplot as plt
# from letta_client import Letta
# import smtplib

# # ---- CONFIG ----
# client = Letta(base_url="http://localhost:8283")  # Connect to local Letta server

# CATEGORY_KEYWORDS = {
#     "Food": ["restaurant", "cafe", "dining", "meal"],
#     "Travel": ["uber", "taxi", "flight", "airbnb", "train"],
#     "Business": ["software", "subscription", "license", "zoom"],
#     "Shopping": ["amazon", "store", "retail", "mall"],
#     "Groceries": ["walmart", "grocery", "supermarket", "costco"],
#     "Other": []
# }

# # ---- FUNCTIONS ----
# def categorize_expense(description):
#     description = str(description).lower()
#     for category, keywords in CATEGORY_KEYWORDS.items():
#         if any(keyword in description for keyword in keywords):
#             return category
#     return "Other"

# def analyze_trends(df):
#     df["Week"] = df["Date"].dt.strftime("%Y-%U")
#     weekly_summary = df.groupby(["Week", "Category"])["Amount"].sum().unstack().fillna(0)
#     return weekly_summary

# def plot_spending(df):
#     fig, ax = plt.subplots(figsize=(10, 6))
#     df.plot(kind="bar", stacked=True, ax=ax)
#     plt.title("Weekly Spending by Category")
#     plt.ylabel("Amount ($)")
#     plt.xlabel("Week")
#     plt.xticks(rotation=45)
#     plt.tight_layout()
#     return fig

# def letta_insights(prompt):
#     try:
#         response = client.complete(prompt=prompt)
#         return response["completion"]
#     except Exception as e:
#         return f"Letta failed to generate insights: {e}"

# def send_email_summary(subject, body, to_email):
#     try:
#         smtp_server = "smtp.gmail.com"
#         smtp_port = 587
#         sender_email = "shreyasdasari12@gmail.com"
#         sender_password = "Shreyas@1698"  # Replace with your actual Gmail app password

#         message = f"Subject: {subject}\n\n{body}"

#         with smtplib.SMTP(smtp_server, smtp_port) as server:
#             server.starttls()
#             server.login(sender_email, sender_password)
#             server.sendmail(sender_email, to_email, message)
#     except Exception as e:
#         st.error(f"Email failed: {e}")

# # ---- STREAMLIT APP ----
# st.set_page_config(page_title="Letta Expense Manager", layout="wide")
# st.title("📅 Letta-Powered Expense Manager")

# uploaded_file = st.file_uploader("Upload your transaction log (CSV)", type=["csv"])

# # ---- EXPENSE ANALYSIS ----
# if uploaded_file:
#     df = pd.read_csv(uploaded_file)
#     df.columns = [col.strip().title() for col in df.columns]

#     if all(col in df.columns for col in ["Date", "Description", "Amount"]):
#         df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
#         df = df.dropna(subset=["Date", "Amount"])
#         df["Category"] = df["Description"].apply(categorize_expense)

#         st.subheader("📊 Categorized Transactions")
#         st.dataframe(df)

#         st.subheader("🔄 Weekly Spending Trends")
#         trends = analyze_trends(df)
#         st.pyplot(plot_spending(trends))

#         this_week = trends.iloc[-1] if not trends.empty else None
#         last_week = trends.iloc[-2] if len(trends) > 1 else None

#         summary_lines = []
#         if this_week is not None and last_week is not None:
#             for category in trends.columns:
#                 change = this_week[category] - last_week[category]
#                 pct = (change / last_week[category]) * 100 if last_week[category] != 0 else 0
#                 if abs(pct) > 20:
#                     insight = f"You spent {pct:.1f}% {'more' if pct > 0 else 'less'} on {category} this week."
#                     st.info(insight)
#                     summary_lines.append(insight)

#         # Generate AI Insights
#         if st.button("Generate Letta Insights"):
#             insight_text = df[["Description", "Amount"]].to_string(index=False)
#             insights = letta_insights(f"Give me smart financial insights based on the following transactions:\n{insight_text}")
#             st.success(insights)
#             summary_lines.append("\nAI Insights:\n" + insights)

#         if summary_lines:
#             full_summary = "\n".join(summary_lines)
#             st.text_area("Weekly Summary", full_summary)
#             recipient_email = st.text_input("Recipient Email")
#             if st.button("Send Summary via Email") and recipient_email:
#                 send_email_summary("Weekly Spending Summary", full_summary, recipient_email)

#         csv = df.to_csv(index=False).encode("utf-8")
#         st.download_button("Download Cleaned Data", csv, "categorized_expenses.csv", "text/csv")

#         st.success("All done! This is Letta automating boring tasks for you 🤖")
#     else:
#         st.error("CSV must contain columns: Date, Description, Amount")
# else:
#     st.info("Upload a CSV file with columns: Date, Description, Amount")

# install letta_client with `pip install letta-client`
from letta_client import Letta

# Connect to Letta Desktop (local server)
client = Letta(base_url="http://localhost:8283")

# Your existing agent ID from Letta Desktop
agent_id = "agent-d432fd25-ab0b-4128-bb13-0ef8f8dac43b"

# Send a message to the agent
response = client.agents.messages.create(
    agent_id=agent_id,
    messages=[
        {
            "role": "user",
            "content": "What are the benefits of using AI in healthcare?"
        }
    ]
)

# ✅ Print usage stats (tokens, steps, etc.)
print("\n📊 Usage Stats:")
print(response.usage)

# ✅ Print each message clearly
print("\n💬 Agent Messages:")
for message in response.messages:
    role = getattr(message, "role", "unknown")
    content = getattr(message, "content", None)

    if isinstance(content, list):
        for item in content:
            if hasattr(item, "text"):
                print(f"{role}: {item.text}")
            else:
                print(f"{role}: {item}")
    else:
        print(f"{role}: {content}")
