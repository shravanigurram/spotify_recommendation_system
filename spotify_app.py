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
        st.write(f"{idx}. 🎵 {rec['track_name']} by {rec['artist']}")

# Function to find similar tracks based on KNN
def find_similar_tracks(track_name, data, model, n_recommendations=5):
    try:
        track_idx = data.index[data["track_name"].str.lower() == track_name.lower()][0]
    except IndexError:
        return []

    distances, indices = model.kneighbors([data_scaled[track_idx]], n_neighbors=n_recommendations + 1)
    recommendations = [
        {'track_name': data.iloc[idx]['track_name'], 'artist': data.iloc[idx]['artist'], 'genre': data.iloc[idx]['genre'], 'distance': distance}
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
        {'track_name': data.iloc[idx]['track_name'], 'artist': data.iloc[idx]['artist'], 'genre': data.iloc[idx]['genre'], 'distance': distance}
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
        {'track_name': data.iloc[idx]['track_name'], 'artist': data.iloc[idx]['artist'], 'genre': data.iloc[idx]['genre'], 'distance': distance}
        for idx, distance in zip(indices.flatten(), distances.flatten()) if idx != track_idx
    ]
    return recommendations[:n_recommendations]

# New function to recommend tracks based on both genre and artist
def recommend_tracks_by_genre_and_artist(genre_name, artist_name, spotify_data, model, features, n_recommendations=20):
    # Filter tracks by the given genre and artist
    filtered_tracks = spotify_data[(spotify_data['artist'] == artist_name) & (spotify_data['genre'] == genre_name)]

    if filtered_tracks.empty:
        st.warning(f"No tracks found for artist: '{artist_name}' and genre: '{genre_name}'")
        return pd.DataFrame()

    # Select a representative track (first track)
    representative_track = filtered_tracks.iloc[0]
    track_features = representative_track[features].values.reshape(1, -1)
    track_features_scaled = scaler.transform(track_features)

    # Find similar tracks using the k-NN model
    distances, indices = model.kneighbors(track_features_scaled, n_neighbors=n_recommendations)

    # Gather recommended tracks (excluding the first one, which is the query track itself)
    recommended_tracks = spotify_data.iloc[indices[0][1:]]
    recommended_tracks['distance'] = distances[0][1:]

    # Return the recommended tracks with their similarity scores
    recommendations = recommended_tracks[['track_name', 'artist', 'genre', 'distance']]
    return recommendations

# Spotify-themed Streamlit App with custom CSS
st.set_page_config(page_title="Spotify Music Recommender", page_icon="🎧", layout="wide")

st.title("🎵 Spotify Music Recommendation System")
st.markdown(
    """
    <style>
    body {
        font-family: 'Arial', sans-serif;
        background-color: #1C1C1C;
        color: white;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #1DB954;
    }
    .stButton>button {
        background-color: #1DB954;
        color: white;
        border-radius: 10px;
        font-size: 18px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #128B31;
        transform: scale(1.05);
    }
    .stTextInput input {
        background-color: #333333;
        border: 1px solid #1DB954;
        color: white;
    }
    .stSlider>div>div>input {
        background-color: #333333;
        color: white;
    }
    .stTextInput input:focus {
        border-color: #1DB954;
    }
    .stTextInput input::placeholder {
        color: #a0a0a0;
    }
    .stWarning {
        background-color: #FF5722;
        color: white;
        border-radius: 5px;
    }
    .stSubheader {
        color: #1DB954;
    }
    .stTab {
        background-color: #333333;
        color: white;
        border-radius: 5px;
    }
    .stTab:hover {
        background-color: #1DB954;
        color: black;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Tab structure for different functionalities
tab1, tab2, tab3, tab4= st.tabs(["Track-based", "Artist-based", "Genre-based", "Genre & Artist-based"])

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

# Artist-based Recommendations
with tab2:
    artist_name = st.text_input("🎨 Enter an Artist Name", placeholder="e.g., Artist_17")
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
    genre_name = st.text_input("🎭 Enter a Genre", placeholder="e.g., Pop")
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
    artist_name = st.text_input("🎨 Enter Artist Name", placeholder="e.g., Artist_17", key="artist_genre_input")
    genre_name = st.text_input("🎭 Enter Genre", placeholder="e.g., Pop", key="genre_genre_input")
    num_recommendations = st.slider("🔢 Number of Recommendations", 1, 20, 5, key="genre_artist_slider")
    if st.button("Get Recommendations based on Artist & Genre 🎶"):
        if artist_name and genre_name:
            recommendations = recommend_tracks_by_genre_and_artist(genre_name, artist_name, spotify_data, knn_model, feature_columns, num_recommendations)
            if not recommendations.empty:
                display_recommendations(f"Recommended Tracks for '{artist_name}' in Genre '{genre_name}':", recommendations.to_dict(orient='records'))
            else:
                st.warning(f"🚫 No tracks found for artist '{artist_name}' in genre '{genre_name}'.")
        else:
            st.warning("⚠️ Please enter both artist name and genre.")

