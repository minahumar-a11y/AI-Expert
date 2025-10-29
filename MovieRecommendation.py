import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
from colorama import init, Fore
import time

# Initialize colorama
init(autoreset=True)

# =======================
# Load and preprocess data
# =======================
def load_data(file_path='imdb_top_1000.csv'):
    try:
        df = pd.read_csv(file_path)
        df['combined_features'] = df['Genre'].fillna('') + ' ' + df['Overview'].fillna('')
        return df
    except FileNotFoundError:
        print(Fore.RED + f"Error: The file '{file_path}' was not found.")
        exit()

movies_df = load_data()

# TF-IDF and cosine similarity
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies_df['combined_features'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# =======================
# Utility Functions
# =======================
def list_genres(df):
    return sorted(set(genre.strip() for sublist in df['Genre'].dropna().str.split(',') for genre in sublist))

genres = list_genres(movies_df)

def processing_animation(text="Processing"):
    print(Fore.BLUE + f"\n{text}", end="", flush=True)
    for _ in range(3):
        print(Fore.YELLOW + ".", end="", flush=True)
        time.sleep(0.5)
    print()

# =======================
# Movie Recommendation
# =======================
def recommend_movies(genre=None, mood=None, rating=None, top_n=5):
    filtered_df = movies_df

    # Apply filters
    if genre:
        filtered_df = filtered_df[filtered_df['Genre'].str.contains(genre, case=False, na=False)]
    if rating:
        filtered_df = filtered_df[filtered_df['IMDB_Rating'] >= rating]

    filtered_df = filtered_df.sample(frac=1).reset_index(drop=True)  # Shuffle

    recommendations = []
    for _, row in filtered_df.iterrows():
        overview = row['Overview']
        if pd.isna(overview):
            continue
        polarity = TextBlob(overview).sentiment.polarity
        if (mood and ((TextBlob(mood).sentiment.polarity < 0 and polarity > 0) or polarity >= 0)) or not mood:
            recommendations.append((row['Series_Title'], polarity))
        if len(recommendations) == top_n:
            break

    return recommendations if recommendations else "No suitable movie recommendations found."

def display_recommendations(recs, name):
    print(Fore.YELLOW + f"\n🎬 AI-Analyzed Movie Recommendations for {name}:")
    for idx, (title, polarity) in enumerate(recs, 1):
        sentiment = "Positive 😍" if polarity > 0 else "Negative 😞" if polarity < 0 else "Neutral 🙂"
        print(f"{Fore.CYAN}{idx}. 🎥 {title} (Polarity: {polarity:.2f}, {sentiment})")

# =======================
# Main AI Assistant Flow
# =======================
def handle_ai(name):
    print(Fore.BLUE + "\n🔍 Let's find the perfect movie for you!\n")

    # Show available genres
    print(Fore.GREEN + "Available Genres: ", end="")
    for idx, genre in enumerate(genres, 1):
        print(f"{Fore.CYAN}{idx}. {genre}", end=", ")
    print("\n")

    # Genre input
    while True:
        genre_input = input(Fore.YELLOW + "Enter genre number or name: ").strip()
        if genre_input.isdigit() and 1 <= int(genre_input) <= len(genres):
            genre = genres[int(genre_input) - 1]
            break
        elif genre_input.title() in genres:
            genre = genre_input.title()
            break
        print(Fore.RED + "Invalid input. Try again.\n")

    # Mood input
    mood = input(Fore.YELLOW + "How do you feel today? (Describe your mood): ").strip()
    processing_animation("Analyzing mood")
    polarity = TextBlob(mood).sentiment.polarity
    mood_desc = "positive 😊" if polarity > 0 else "negative 😞" if polarity < 0 else "neutral 😐"
    print(f"\n{Fore.GREEN}Your mood is {mood_desc} (Polarity: {polarity:.2f}).\n")

    # Rating input
    while True:
        rating_input = input(Fore.YELLOW + "Enter minimum IMDb rating (7.6–9.3) or 'skip': ").strip()
        if rating_input.lower() == 'skip':
            rating = None
            break
        try:
            rating = float(rating_input)
            if 7.6 <= rating <= 9.3:
                break
            print(Fore.RED + "Rating out of range. Try again.\n")
        except ValueError:
            print(Fore.RED + "Invalid input. Try again.\n")

    # Get Recommendations
    processing_animation(f"Finding movies for {name}")
    recs = recommend_movies(genre=genre, mood=mood, rating=rating, top_n=5)
    if isinstance(recs, str):
        print(Fore.RED + recs + "\n")
    else:
        display_recommendations(recs, name)

    # Ask user for more recommendations
    while True:
        action = input(Fore.YELLOW + "\nWould you like more recommendations? (yes/no): ").strip().lower()
        if action == 'no':
            print(Fore.GREEN + f"\nEnjoy your movie picks, {name}! 🎬🍿\n")
            break
        elif action == 'yes':
            recs = recommend_movies(genre=genre, mood=mood, rating=rating, top_n=5)
            if isinstance(recs, str):
                print(Fore.RED + recs + "\n")
            else:
                display_recommendations(recs, name)
        else:
            print(Fore.RED + "Invalid choice. Try again.\n")

# =======================
# Main Program
# =======================
def main():
    print(Fore.BLUE + "🎥 Welcome to your Personal Movie Recommendation Assistant! 🎬\n")
    name = input(Fore.YELLOW + "What's your name? ").strip()
    print(f"\n{Fore.GREEN}Great to meet you, {name}!\n")
    handle_ai(name)

if __name__ == "__main__":
    main()
