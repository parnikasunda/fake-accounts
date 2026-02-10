import streamlit as st
import pandas as pd
import pickle

# Load model
model = pickle.load(open("model.pkl", "rb"))

st.title("Fake Account Detection App")
st.write(
    "This app predicts whether a social media account is fake or genuine "
    "based on behavioral features using a trained machine learning model."
)


# --- Demo presets ---
genuine_demo = {
    "followers": 5000.0,
    "friends": 200.0,
    "favorites": 2500.0,
    "verified": 1,
    "desc_length": 120,
    "word_count": 20,
    "empty_desc": 0,
    "lang_encoded": 1
}

fake_demo = {
    "followers": 0.0,
    "friends": 5000.0,
    "favorites": 0.0,
    "verified": 0,
    "desc_length": 0,
    "word_count": 0,
    "empty_desc": 1,
    "lang_encoded": 0
}


# --- Demo buttons ---
col1, col2 = st.columns(2)

if col1.button("Load Genuine Demo"):
    st.session_state.update(genuine_demo)

if col2.button("Load Fake Demo"):
    st.session_state.update(fake_demo)

st.write("Enter account details:")

followers = st.number_input(
    "Followers",
    value=st.session_state.get("followers", 0.0)
)

friends = st.number_input(
    "Friends",
    value=st.session_state.get("friends", 0.0)
)

favorites = st.number_input(
    "Favorites",
    value=st.session_state.get("favorites", 0.0)
)

verified = st.selectbox(
    "Verified Account?",
    [0, 1],
    index=st.session_state.get("verified", 0)
)

desc_length = st.number_input(
    "Description Length",
    value=st.session_state.get("desc_length", 0)
)

word_count = st.number_input(
    "Word Count",
    value=st.session_state.get("word_count", 0)
)

empty_desc = st.selectbox(
    "Empty Description?",
    [0, 1],
    index=st.session_state.get("empty_desc", 0)
)

lang_encoded = st.number_input(
    "Language Code",
    value=st.session_state.get("lang_encoded", 0)
)

if st.button("Predict"):

    follower_friend_ratio = followers / (friends + 1)
    engagement_score = favorites / (followers + 1)
    social_balance = abs(followers - friends)

    input_data = pd.DataFrame([[
        followers,
        friends,
        favorites,
        verified,
        follower_friend_ratio,
        engagement_score,
        social_balance,
        desc_length,
        word_count,
        empty_desc,
        lang_encoded
    ]], columns=[
        "followers_scaled",
        "friends_scaled",
        "favourites_scaled",
        "verified",
        "follower_friend_ratio",
        "engagement_score",
        "social_balance",
        "desc_length",
        "word_count",
        "empty_desc",
        "lang_encoded"
    ])

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.error("Prediction: Fake Account")
    else:
        st.success("Prediction: Genuine Account")
