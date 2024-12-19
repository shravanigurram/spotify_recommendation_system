import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# Load KNN model, dataset, and scaler
with open('knn.pkl', 'rb') as file:
    knn_model = pickle.load(file)

DATA_PATH = 'spotify_synthetic_data.csv'
spotify_data = pd.read_csv(DATA_PATH)

# Assuming features and scaler are pre-defined
feature_columns = ["danceability", "energy", "loudness", "speechiness", "acousticness", "instrumentalness", "liveness", "valence", "tempo"]
scaler = StandardScaler()
data_scaled = scaler.fit_transform(spotify_data[feature_columns])

# Function to format recommendations
def display_recommendations(title, recommendations):
    st.subheader(title)
    for idx, rec in enumerate(recommendations, 1):
        st.write(f"{idx}. 🎵 {rec['track_name']}")

# Function to find similar tracks based on KNN
# Function to find similar tracks based on KNN
def find_similar_tracks(track_name, data, model, n_recommendations=5):
    try:
        track_idx = data.index[data["track_name"].str.lower() == track_name.lower()][0]
    except IndexError:
        return []

    distances, indices = model.kneighbors([data_scaled[track_idx]], n_neighbors=n_recommendations + 1)
    recommendations = []
    for idx, distance in zip(indices.flatten(), distances.flatten()):
        if idx != track_idx:
            recommendations.append({
                'track_name': data.iloc[idx]['track_name'],
                'distance': distance
            })
    return recommendations[:n_recommendations]  # Return exactly n_recommendations

# Function to find recommendations by artist
def recommend_by_artist(artist_name, data, model, n_recommendations=5):
    artist_tracks_idx = data.index[data['artist'].str.lower() == artist_name.lower()].tolist()
    if not artist_tracks_idx:
        return []

    track_idx = artist_tracks_idx[0]
    distances, indices = model.kneighbors([data_scaled[track_idx]], n_neighbors=n_recommendations + 1)
    recommendations = []
    for idx, distance in zip(indices.flatten(), distances.flatten()):
        if idx != track_idx:
            recommendations.append({
                'track_name': data.iloc[idx]['track_name'],
                'distance': distance
            })
    return recommendations[:n_recommendations]  # Return exactly n_recommendations

# Function to find recommendations by genre
def recommend_by_genre(genre_name, data, model, n_recommendations=5):
    genre_tracks_idx = data.index[data['genre'].str.lower() == genre_name.lower()].tolist()
    if not genre_tracks_idx:
        return []

    track_idx = genre_tracks_idx[0]
    distances, indices = model.kneighbors([data_scaled[track_idx]], n_neighbors=n_recommendations + 1)
    recommendations = []
    for idx, distance in zip(indices.flatten(), distances.flatten()):
        if idx != track_idx:
            recommendations.append({
                'track_name': data.iloc[idx]['track_name'],
                'distance': distance
            })
    return recommendations[:n_recommendations]  # Return exactly n_recommendations


# Spotify-themed Streamlit App
st.title("🎵 Spotify Recommendation System")
st.markdown(
    """
    <style>
    body, .stApp {
        font-family: Arial, sans-serif;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #1DB954;
    }
    .stButton>button {
        background-color: #1DB954;
        color: white;
        border-radius: 10px;
        font-size: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Tab structure for different functionalities
tab1, tab2, tab3 = st.tabs(["Track-based", "Artist-based", "Genre-based"])

# Track-based Recommendations
with tab1:
    track_name = st.text_input("🎧 Enter a Track Name", placeholder="e.g., Track_1")
    num_recommendations = st.slider("🔢 Number of Recommendations", 1, 20, 5)
    if st.button("Get Recommendations based on track🎶"):
        if track_name:
            recommendations = find_similar_tracks(track_name, spotify_data, knn_model, num_recommendations)
            if recommendations:
                display_recommendations(f"Recommended Tracks for '{track_name}':", recommendations)
            else:
                st.warning("🚫 Track not found in the dataset.")
        else:
            st.warning("⚠️ Please enter a track name.")

# Artist-based Recommendations
with tab2:
    artist_name = st.text_input("🎨 Enter an Artist Name", placeholder="e.g., Artist_17")
    num_recommendations = st.slider("🔢 Number of Recommendations", 1, 20, 5, key="artist_slider")
    if st.button("Get Recommendations based on Artist🎶"):
        if artist_name:
            recommendations = recommend_by_artist(artist_name, spotify_data, knn_model, num_recommendations)
            if recommendations:
                display_recommendations(f"Recommended Tracks for Artist '{artist_name}':", recommendations)
            else:
                st.warning(f"🚫 Artist '{artist_name}' not found in the dataset.")
        else:
            st.warning("⚠️ Please enter an artist name.")

# Genre-based Recommendations
with tab3:
    genre_name = st.text_input("🎭 Enter a Genre", placeholder="e.g., Pop")
    num_recommendations = st.slider("🔢 Number of Recommendations", 1, 20, 5, key="genre_slider")
    if st.button("Get Recommendations based on Genre🎶"):
        if genre_name:
            recommendations = recommend_by_genre(genre_name, spotify_data, knn_model, num_recommendations)
            if recommendations:
                display_recommendations(f"Recommended Tracks for Genre '{genre_name}':", recommendations)
            else:
                st.warning(f"🚫 Genre '{genre_name}' not found in the dataset.")
        else:
            st.warning("⚠️ Please enter a genre.")
