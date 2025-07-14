import streamlit as st
import pandas as pd
import joblib

# --- Load data and models ---
@st.cache_data
def load_data():
    df = pd.read_csv("..\Data\zomato_clean_processed.csv")
    df['City'] = df['City'].astype(str).str.strip().str.lower()
    df['Cuisine'] = df['Cuisine'].astype(str).str.strip().str.lower()
    return df

@st.cache_resource
def load_models():
    price_model = joblib.load("..\Models\price_predictor_rf.pkl")
    price_scaler = joblib.load("..\Models\price_scaler.pkl")
    return price_model, price_scaler

df = load_data()
price_model, price_scaler = load_models()

# --- UI Header ---
st.set_page_config(page_title="🍽️ ZomatoX ML App", layout="wide")
st.title("🍽️ ZomatoX – Restaurant Intelligence Platform")
st.markdown("Predict menu prices or explore top restaurants by cuisine and city.")

st.sidebar.title("🔍 Select Mode")
mode = st.sidebar.radio("Choose what you want to do:", ["📌 Recommend Restaurants", "💡 Predict Menu Item Price"])

# --- Mode 1: Recommend Restaurants ---
if mode == "📌 Recommend Restaurants":
    st.header("📌 Restaurant Recommender")

    city_input = st.selectbox("Select City", sorted(df['City'].unique()))
    cuisine_input = st.selectbox("Select Cuisine", sorted(df['Cuisine'].unique()))
    sort_by = st.radio("Sort By", ["Average Rating", "Price (Low to High)", "Best Value"], horizontal=True)

    # Filter
    filtered_df = df[(df['City'] == city_input) & (df['Cuisine'] == cuisine_input)]
    
    if filtered_df.empty:
        st.warning("❌ No restaurants found for the selected city and cuisine.")
    else:
        if sort_by == "Average Rating":
            filtered_df = filtered_df.sort_values(by="Average_Rating", ascending=False)
        elif sort_by == "Price (Low to High)":
            filtered_df = filtered_df.sort_values(by="Prices")
        else:
            filtered_df = filtered_df.sort_values(by="Price_per_Vote")

        st.success(f"Found {len(filtered_df)} restaurants")
        st.dataframe(filtered_df[['Restaurant_Name', 'Place_Name', 'Prices', 'Average_Rating', 'Votes']].head(10))

# --- Mode 2: Predict Menu Item Price ---
elif mode == "💡 Predict Menu Item Price":
    st.header("💰 Menu Price Predictor")
    st.markdown("Input restaurant and menu features to predict the expected price (INR).")

    col1, col2, col3 = st.columns(3)

    with col1:
        dining_rating = st.slider("Dining Rating", 0.0, 5.0, 4.2, 0.1)
        delivery_rating = st.slider("Delivery Rating", 0.0, 5.0, 4.0, 0.1)
        dining_votes = st.number_input("Dining Votes", 0, 10000, 120)
        delivery_votes = st.number_input("Delivery Votes", 0, 10000, 95)
        votes = st.number_input("Total Votes", 0, 20000, 215)

    with col2:
        avg_rating = st.slider("Average Rating", 0.0, 5.0, 4.1, 0.1)
        price_per_vote = st.number_input("Price per Vote", 0.0, 100.0, 2.5)
        log_price = st.number_input("Log Price", 1.0, 6.0, 4.0)
        is_bestseller = st.selectbox("Is Bestseller?", ["Yes", "No"])
        is_expensive = st.selectbox("Is Expensive?", ["Yes", "No"])

    with col3:
        rest_popularity = st.number_input("Restaurant Popularity (item count)", 0, 100, 8)
        avg_rating_rest = st.slider("Avg Rating of Restaurant", 0.0, 5.0, 4.1, 0.1)
        avg_price_rest = st.number_input("Avg Price of Restaurant", 0.0, 1000.0, 250.0)
        avg_rating_cuisine = st.slider("Avg Rating of Cuisine", 0.0, 5.0, 4.0, 0.1)
        avg_price_cuisine = st.number_input("Avg Price of Cuisine", 0.0, 1000.0, 300.0)

    # Features in correct order
    input_features = [[
        dining_rating, delivery_rating, dining_votes, delivery_votes, votes,
        avg_rating, votes, price_per_vote, log_price,
        1 if is_bestseller == "Yes" else 0,
        rest_popularity, avg_rating_rest, avg_price_rest,
        avg_rating_cuisine, avg_price_cuisine,
        avg_rating_cuisine, avg_price_cuisine,
        1 if avg_rating >= 4.0 else 0,
        1 if is_expensive == "Yes" else 0
    ]]

    scaled_input = price_scaler.transform(input_features)
    prediction = price_model.predict(scaled_input)[0]

    st.markdown("---")
    st.subheader("💰 Predicted Menu Price:")
    st.success(f"₹ {prediction:.2f}")

