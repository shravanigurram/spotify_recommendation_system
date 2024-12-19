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

# Function to find recommendations by artist
def recommend_by_artist(artist_name, data, model, n_recommendations=5):
    artist_tracks_idx = data.index[data['artist'].str.lower() == artist_name.lower()].tolist()
    if not artist_tracks_idx:
        return []

    track_idx = artist_tracks_idx[0]
    distances, indices = model.kneighbors([data_scaled[track_idx]], n_neighbors=n_recommendations + 1)
    recommendations = [
        {'track_name': data.iloc[idx]['track_name'], 'distance': distance}
        for idx, distance in zip(indices.flatten(), distances.flatten()) if idx != track_idx
    ]
    return recommendations[:n_recommendations]

# Function to find recommendations by genre
def recommend_by_genre(genre_name, data, model, n_recommendations=5):
    genre_tracks_idx = data.index[data['genre'].str.lower() == genre_name.lower()].tolist()
    if not genre_tracks_idx:
        return []

    track_idx = genre_tracks_idx[0]
    distances, indices = model.kneighbors([data_scaled[track_idx]], n_neighbors=n_recommendations + 1)
    recommendations = [
        {'track_name': data.iloc[idx]['track_name'], 'distance': distance}
        for idx, distance in zip(indices.flatten(), distances.flatten()) if idx != track_idx
    ]
    return recommendations[:n_recommendations]

# Function to recommend tracks based on both genre and artist
def recommend_by_genre_and_artist(genre_name, artist_name, data, model, data_scaled, n_recommendations=5):
    # Filter tracks by genre and artist
    genre_artist_tracks_idx = data.index[
        (data['genre'].str.lower() == genre_name.lower()) & (data['artist_name'].str.lower() == artist_name.lower())
    ].tolist()
    
    if not genre_artist_tracks_idx:
        return []

    track_idx = genre_artist_tracks_idx[0]
    distances, indices = model.kneighbors([data_scaled[track_idx]], n_neighbors=n_recommendations + 1)  
    
    # Create recommendations list excluding the input track
    recommendations = [
        {'track_name': data.iloc[idx]['track_name'], 'distance': distance}
        for idx, distance in zip(indices.flatten(), distances.flatten()) if idx != track_idx
    ]
    
    return recommendations[:n_recommendations]

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
    track_name = st.text_input("🎧 Enter a Track Name", placeholder="e.g., Track_1", key="track_name")
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

# Artist-based Recommendations
with tab2:
    artist_name = st.text_input("🎨 Enter an Artist Name", placeholder="e.g., Artist_17", key="artist_name")
    num_recommendations = st.slider("🔢 Number of Recommendations", 1, 20, 5, key="artist_slider")
    if st.button("Get Recommendations based on Artist 🎶"):
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
    genre_name = st.text_input("🎭 Enter a Genre", placeholder="e.g., Pop", key="genre_name")
    num_recommendations = st.slider("🔢 Number of Recommendations", 1, 20, 5, key="genre_slider")
    if st.button("Get Recommendations based on Genre 🎶"):
        if genre_name:
            recommendations = recommend_by_genre(genre_name, spotify_data, knn_model, num_recommendations)
            if recommendations:
                display_recommendations(f"Recommended Tracks for Genre '{genre_name}':", recommendations)
            else:
                st.warning(f"🚫 Genre '{genre_name}' not found in the dataset.")
        else:
            st.warning("⚠️ Please enter a genre.")

# Genre & Artist-based Recommendations
with tab4:
    st.title("🎶 Track Recommendations based on Genre & Artist")

    # Input fields for genre and artist
    genre_name = st.text_input("🎭 Enter a Genre", placeholder="e.g., Pop", key="genre_artist")
    artist_name = st.text_input("🎤 Enter an Artist", placeholder="e.g., Artist_1", key="artist_genre")

    # Slider to select the number of recommendations
    num_recommendations = st.slider("🔢 Number of Recommendations", 1, 20, 5, key="genre_artist_slider")

    # Button to trigger the recommendation process
    if st.button("Get Recommendations based on Genre & Artist 🎶"):
        if genre_name and artist_name:
            # Call the function to get combined recommendations
            recommendations = recommend_by_genre_and_artist(genre_name, artist_name, spotify_data, knn_model, data_scaled, num_recommendations)
            
            # Display recommendations
            if recommendations:
                st.write(f"**Recommended Tracks for Genre '{genre_name}' and Artist '{artist_name}':**")
                for rec in recommendations:
                    st.write(f"- **{rec['track_name']}** (Distance: {rec['distance']:.2f})")
            else:
                st.warning(f"🚫 No tracks found for genre '{genre_name}' and artist '{artist_name}' in the dataset.")
        else:
            st.warning("⚠️ Please enter both a genre and an artist.")
