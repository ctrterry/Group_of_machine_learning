from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

with open("rf_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        genre_doc = int("Genre_Documentary" in request.form.getlist("genres"))
        genre_anim = int("Genre_Animation" in request.form.getlist("genres"))
        genre_drama = int("Genre_Drama" in request.form.getlist("genres"))
        genre_horror = int("Genre_Horror" in request.form.getlist("genres"))

        director_likes = float(request.form["director_facebook_likes"])
        actor3_likes = float(request.form["actor_3_facebook_likes"])
        movie_likes = float(request.form["movie_facebook_likes"])
        budget = float(request.form["budget_adjusted"])
        runtime = float(request.form["runtime"])
        num_user_reviews = float(request.form["num_user_for_reviews"])
        num_critic_reviews = float(request.form["num_critic_for_reviews"])
        release_year = int(request.form["release_year"])

        features = [
            genre_doc, genre_anim, director_likes, actor3_likes,
            genre_drama, movie_likes, genre_horror, budget, runtime,
            num_user_reviews, release_year, num_critic_reviews
        ]
        prediction = round(model.predict([features])[0], 2)

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
