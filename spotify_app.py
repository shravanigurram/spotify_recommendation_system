import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# Load KNN model, dataset, and scaler
with open('knn.pkl', 'rb') as file:
    knn_model = pickle.load(file)

DATA_PATH = 'spotify_dataset.csv'
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
def find_similar_tracks(track_name, data, model, n_recommendations=5):
    try:
        track_idx = data.index[data["track_name"].str.lower() == track_name.lower()][0]
    except IndexError:
        return []

    distances, indices = model.kneighbors([data_scaled[track_idx]], n_neighbors=n_recommendations + 1)
    recommendations = [
        {'track_name': data.iloc[idx]['track_name'], 'distance': distance}
        for idx, distance in zip(indices.flatten(), distances.flatten()) if idx != track_idx
    ]
    return recommendations[:n_recommendations]

# Function to recommend tracks by genre and artist
def recommend_tracks_by_genre_and_artist(genre_name, artist_name, data, model, features, n_recommendations=20):
    filtered_tracks = data[(data['artist'].str.lower() == artist_name.lower()) & (data['genre'].str.lower() == genre_name.lower())]
    if filtered_tracks.empty:
        return pd.DataFrame()

    representative_track = filtered_tracks.iloc[0]
    track_features = representative_track[features].values.reshape(1, -1)
    track_features_scaled = scaler.transform(track_features)

    distances, indices = model.kneighbors(track_features_scaled, n_neighbors=n_recommendations)
    recommended_tracks = data.iloc[indices[0][1:]]
    recommended_tracks['distance'] = distances[0][1:]
    return recommended_tracks[['track_name', 'artist', 'genre', 'distance']]

# Spotify-themed Streamlit App
st.set_page_config(page_title="Spotify Music Recommender", page_icon="🎧", layout="wide")

st.title("🎵 Spotify Music Recommendation System")
st.markdown(
    """
    <style>
    body, .stApp {
        font-family: 'Trebuchet MS', sans-serif;
        background-color: #f7f7f7;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #1DB954;
    }
    .stButton>button {
        background-color: #1DB954;
        color: white;
        border-radius: 5px;
        font-size: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Tab structure for different functionalities
tab1, tab2, tab3, tab4 = st.tabs(["Track-based", "Artist-based", "Genre-based", "Genre & Artist-based"])

# Track-based Recommendations
with tab1:
    track_name = st.text_input("🎧 Enter a Track Name", placeholder="e.g., Track_1")
    num_recommendations = st.slider("🔢 Number of Recommendations", 1, 20, 5)
    if st.button("Get Recommendations based on Track 🎶"):
        if track_name:
            recommendations = find_similar_tracks(track_name, spotify_data, knn_model, num_recommendations)
            if recommendations:
                display_recommendations(f"Recommended Tracks for '{track_name}':", recommendations)
            else:
                st.warning("🚫 Track not found in the dataset.")
        else:
            st.warning("⚠️ Please enter a track name.")

# Genre & Artist-based Recommendations
with tab4:
    genre_name = st.text_input("🎭 Enter a Genre", placeholder="e.g., Pop", key="genre_artist_input")
    artist_name = st.text_input("🎨 Enter an Artist Name", placeholder="e.g., Artist_17", key="artist_genre_input")
    num_recommendations = st.slider("🔢 Number of Recommendations", 1, 20, 5, key="genre_artist_slider")
    if st.button("Get Recommendations based on Genre & Artist 🎶"):
        if genre_name and artist_name:
            recommendations = recommend_tracks_by_genre_and_artist(genre_name, artist_name, spotify_data, knn_model, feature_columns, num_recommendations)
            if not recommendations.empty:
                st.dataframe(recommendations)
            else:
                st.warning(f"🚫 No tracks found for artist '{artist_name}' and genre '{genre_name}'.")
        else:
            st.warning("⚠️ Please enter both genre and artist.")
