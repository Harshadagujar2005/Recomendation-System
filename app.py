from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# -------------------------------
# 🎬 Small built-in dataset
# -------------------------------
movies = {
    'movieId': [1, 2, 3, 4, 5],
    'title': ['Toy Story (1995)', 'Jumanji (1995)', 'Heat (1995)',
              'GoldenEye (1995)', 'Batman Forever (1995)'],
    'genres': ['Animation|Children|Comedy', 'Adventure|Children|Fantasy',
               'Action|Crime|Thriller', 'Action|Adventure|Thriller',
               'Action|Adventure|Comedy']
}
movies_df = pd.DataFrame(movies)

# -------------------------------
# 🔹 Content-based similarity
# -------------------------------
tfidf = TfidfVectorizer(stop_words='english')
movies_df['genres'] = movies_df['genres'].fillna('')
tfidf_matrix = tfidf.fit_transform(movies_df['genres'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def content_recommend(movie_title, df, similarity_matrix, top_n=3):
    matches = df[df['title'].str.lower().str.contains(movie_title.lower(), regex=False)]
    if matches.empty:
        return []
    idx = matches.index[0]
    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    movie_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[movie_indices].tolist()


# -------------------------------
# 🌐 Flask route
# -------------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    recommendations = []
    if request.method == "POST":
        movie = request.form.get("movie")
        if movie:
            recommendations = content_recommend(movie, movies_df, cosine_sim, top_n=3)

    return render_template("index.html", recommendations=recommendations)

if __name__ == "__main__":
    app.run(debug=True)
