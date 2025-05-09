

# CREATE TABLES FOR RAW IMDB DATA
```.open C:/Users/sasaa/OneDrive/Documents/GOLANG/src/MyVault/NOTES/UC-Davis/S25/ECS171/PROJECT/db/imdb.duckdb```

```
CREATE TABLE IF NOT EXISTS name_basics (
    nconst STRING,
    primaryName STRING,
    birthYear INTEGER,
    deathYear INTEGER,
    primaryProfession STRING,
    knownForTitles STRING
);
```

```
INSERT INTO name_basics
SELECT *
FROM read_csv('C:/Users/sasaa/OneDrive/Documents/GOLANG/src/MyVault/NOTES/UC-Davis/S25/ECS171/PROJECT/data/name.basics.tsv', delim='\t', header=True,  nullstr='\N', columns={'nconst': 'STRING', 'primaryName': 'STRING', 'birthYear': 'INTEGER', 'deathYear': 'INTEGER', 'primaryProfession': 'STRING', 'knownForTitles': 'STRING'});
```

```
CREATE TABLE IF NOT EXISTS title_basics (
    tconst STRING,
    titleType STRING,
    primaryTitle STRING,
    originalTitle STRING,
    isAdult INTEGER,
    startYear INTEGER,
    endYear INTEGER,
    runtimeMinutes INTEGER,
    genres STRING
);
```

```
INSERT INTO title_basics
SELECT *
FROM read_csv('C:/Users/sasaa/OneDrive/Documents/GOLANG/src/MyVault/NOTES/UC-Davis/S25/ECS171/PROJECT/data/title.basics.tsv', delim='\t', header=True, nullstr='\N', columns={'tconst': 'STRING', 'titleType': 'STRING', 'primaryTitle': 'STRING', 'originalTitle': 'STRING', 'isAdult': 'INTEGER', 'startYear': 'INTEGER', 'endYear': 'INTEGER', 'runtimeMinutes': 'INTEGER', 'genres': 'STRING'});
```

```
CREATE TABLE IF NOT EXISTS title_akas (
    titleId STRING,
    ordering INTEGER,
    title STRING,
    region STRING,
    language STRING,
    types STRING,
    attributes STRING,
    isOriginalTitle INTEGER
);
```

```
INSERT INTO title_akas
SELECT *
FROM read_csv('C:/Users/sasaa/OneDrive/Documents/GOLANG/src/MyVault/NOTES/UC-Davis/S25/ECS171/PROJECT/data/title.akas.tsv', delim='\t', header=True, nullstr='\N', columns={'titleId': 'STRING', 'ordering': 'INTEGER', 'title': 'STRING', 'region': 'STRING', 'language': 'STRING', 'types': 'STRING', 'attributes': 'STRING', 'isOriginalTitle': 'INTEGER'});
```

```
CREATE TABLE IF NOT EXISTS title_episode (
    tconst STRING,
    parentTconst STRING,
    seasonNumber INTEGER,
    episodeNumber INTEGER
);
```

```
INSERT INTO title_episode
SELECT *
FROM read_csv('C:/Users/sasaa/OneDrive/Documents/GOLANG/src/MyVault/NOTES/UC-Davis/S25/ECS171/PROJECT/data/title.episode.tsv', delim='\t', header=True, nullstr='\N', columns={'tconst': 'STRING', 'parentTconst': 'STRING', 'seasonNumber': 'INTEGER', 'episodeNumber': 'INTEGER'});
```
```CREATE TABLE IF NOT EXISTS title_episode (
    tconst STRING,
    parentTconst STRING,
    seasonNumber INTEGER,
    episodeNumber INTEGER
);
```

```
INSERT INTO title_episode
SELECT *
FROM read_csv('C:/Users/sasaa/OneDrive/Documents/GOLANG/src/MyVault/NOTES/UC-Davis/S25/ECS171/PROJECT/data/title.episode.tsv', delim='\t', header=True, nullstr='\N', columns={'tconst': 'STRING', 'parentTconst': 'STRING', 'seasonNumber': 'INTEGER', 'episodeNumber': 'INTEGER'});
```

```
CREATE TABLE IF NOT EXISTS title_principals (
    tconst STRING,
    ordering INTEGER,
    nconst STRING,
    category STRING,
    job STRING,
    characters STRING
);
```

```
INSERT INTO title_principals
SELECT *
FROM read_csv('C:/Users/sasaa/OneDrive/Documents/GOLANG/src/MyVault/NOTES/UC-Davis/S25/ECS171/PROJECT/data/title.principals.tsv', delim='\t', header=True, nullstr='\N', columns={'tconst': 'STRING', 'ordering': 'INTEGER', 'nconst': 'STRING', 'category': 'STRING', 'job': 'STRING', 'characters': 'STRING'});
```

```
CREATE TABLE IF NOT EXISTS title_ratings (
    tconst STRING,
    averageRating FLOAT,
    numVotes INTEGER
);
```

```
INSERT INTO title_ratings
SELECT *
FROM read_csv('C:/Users/sasaa/OneDrive/Documents/GOLANG/src/MyVault/NOTES/UC-Davis/S25/ECS171/PROJECT/data/title.ratings.tsv', delim='\t', header=True, nullstr='\N', columns={'tconst': 'STRING', 'averageRating': 'FLOAT', 'numVotes': 'INTEGER'});
```

# PRINT MOVIES PER ACTOR
```
SELECT
    n.primaryName AS actor_name,
    COUNT(tp.tconst) AS movie_appearances
FROM name_basics n
JOIN title_principals tp ON n.nconst = tp.nconst
JOIN title_basics tb ON tp.tconst = tb.tconst
WHERE (tp.category = 'actor' OR tp.category = 'actress')
  AND tb.titleType = 'movie'
GROUP BY n.nconst, n.primaryName
ORDER BY movie_appearances DESC;
```

# AVG. RATING PER ACTOR
```SELECT 
    n.nconst AS actor_tconst,
    AVG(tr.averageRating) AS avg_movie_rating
FROM name_basics n
JOIN title_principals tp ON n.nconst = tp.nconst
JOIN title_basics tb ON tp.tconst = tb.tconst
JOIN title_ratings tr ON tb.tconst = tr.tconst
WHERE (tp.category = 'actor' OR tp.category = 'actress')
  AND tb.titleType = 'movie'
GROUP BY n.nconst
HAVING COUNT(tr.tconst) > 0
ORDER BY avg_movie_rating DESC
LIMIT 10;
```

# Goal is to have
actor | avg_movie_rating | avg_votes_per_movie | total_movies | rating_std_dev|high_rated_movie_count| career_span| recent_avg_rating|


# MOVIES IN LAST 5-YEARS
```
CREATE TEMPORARY TABLE actor_scores_filtered AS
SELECT *
FROM read_csv('C:/Users/sasaa/OneDrive/Documents/GOLANG/src/MyVault/NOTES/UC-Davis/S25/ECS171/PROJECT/actor_scores_filtered.csv', delim=',', header=True);
```

```
CREATE TABLE IF NOT EXISTS movie_actor_features_2020_2025 AS
SELECT 
    tp.tconst AS tconst,
    AVG(asf.avg_movie_rating) AS cast_avg_movie_rating,
    AVG(asf.avg_votes_per_movie) AS cast_avg_votes_per_movie,
    AVG(asf.total_movies) AS cast_avg_total_movies,
    AVG(asf.rating_std_dev) AS cast_avg_rating_std_dev,
    AVG(asf.high_rated_movie_count) AS cast_avg_high_rated_movie_count,
    AVG(asf.career_span) AS cast_avg_career_span,
    AVG(asf.recent_avg_rating) AS cast_avg_recent_avg_rating,
    tr.averageRating AS movie_rating
FROM title_principals tp
JOIN title_basics tb ON tp.tconst = tb.tconst
JOIN title_ratings tr ON tb.tconst = tr.tconst
JOIN actor_scores_filtered asf ON tp.nconst = asf.actor
WHERE (tp.category = 'actor' OR tp.category = 'actress')
  AND tb.titleType = 'movie'
  AND tb.startYear >= 2020
  AND tb.startYear <= 2025
GROUP BY tp.tconst, tr.averageRating;
```

```
COPY movie_actor_features_2020_2025 TO 'C:/Users/sasaa/OneDrive/Documents/GOLANG/src/MyVault/NOTES/UC-Davis/S25/ECS171/PROJECT/movie_actor_features_2020_2025.csv' WITH (HEADER, DELIMITER ',');
```

# ACTOR SCORES
## Step 1: Create a temporary table for actor-movie pairs
```sql
CREATE TEMPORARY TABLE actor_movies_temp AS
SELECT 
    n.nconst AS actor_tconst,
    tr.averageRating,
    tr.numVotes,
    tb.startYear
FROM name_basics n
JOIN title_principals tp ON n.nconst = tp.nconst
JOIN title_basics tb ON tp.tconst = tb.tconst
LEFT JOIN title_ratings tr ON tb.tconst = tr.tconst
WHERE (tp.category = 'actor' OR tp.category = 'actress')
  AND tb.titleType = 'movie'
  AND tb.startYear IS NOT NULL;
```


## Step 2: Compute career span and filter out unrealistic values
```sql
CREATE TEMPORARY TABLE actor_career_span_temp AS
SELECT 
    actor_tconst,
    (MAX(startYear) - MIN(startYear)) AS career_span
FROM actor_movies_temp
GROUP BY actor_tconst
HAVING (MAX(startYear) - MIN(startYear)) <= 80;
```

## Step 3: Compute aggregate features
```sql
CREATE TEMPORARY TABLE actor_ratings_temp AS
SELECT 
    amt.actor_tconst,
    AVG(amt.averageRating) AS avg_movie_rating,
    AVG(amt.numVotes) AS avg_votes_per_movie,
    STDDEV(amt.averageRating) AS rating_std_dev,
    SUM(CASE WHEN amt.averageRating >= 8.0 THEN 1 ELSE 0 END) AS high_rated_movie_count
FROM actor_movies_temp amt
JOIN actor_career_span_temp acst ON amt.actor_tconst = acst.actor_tconst
GROUP BY amt.actor_tconst
HAVING COUNT(amt.averageRating) > 0;
```

```sql
CREATE TEMPORARY TABLE actor_total_movies_temp AS
SELECT 
    amt.actor_tconst,
    COUNT(*) AS total_movies
FROM actor_movies_temp amt
JOIN actor_career_span_temp acst ON amt.actor_tconst = acst.actor_tconst
GROUP BY amt.actor_tconst;
```

```sql
CREATE TEMPORARY TABLE actor_recent_rating_temp AS
SELECT 
    amt.actor_tconst,
    AVG(amt.averageRating) AS recent_avg_rating
FROM actor_movies_temp amt
JOIN actor_career_span_temp acst ON amt.actor_tconst = acst.actor_tconst
WHERE amt.startYear >= 2020
GROUP BY amt.actor_tconst;
```

## Step 4: Create the final actor_scores table
```sql
CREATE TABLE IF NOT EXISTS actor_scores AS
SELECT 
    art.actor_tconst AS actor,
    art.avg_movie_rating,
    art.avg_votes_per_movie,
    tmt.total_movies,
    art.rating_std_dev,
    art.high_rated_movie_count,
    cst.career_span,
    rrt.recent_avg_rating
FROM actor_ratings_temp art
JOIN actor_total_movies_temp tmt ON art.actor_tconst = tmt.actor_tconst
JOIN actor_career_span_temp cst ON art.actor_tconst = cst.actor_tconst
LEFT JOIN actor_recent_rating_temp rrt ON art.actor_tconst = rrt.actor_tconst;
```

## Export the updated table
```sql
COPY actor_scores TO 'C:/Users/sasaa/OneDrive/Documents/GOLANG/src/MyVault/NOTES/UC-Davis/S25/ECS171/PROJECT/actor_scores_updated.csv' WITH (HEADER, DELIMITER ',');
```


# PRUNING
- There were 142 actors with carrer span over 80 years, most of those actors started at the beginning of the 20th century and had a missing year od death
- There are e.g., 650,000 with 1 movie, 110,000 with 2 movies, for those actors the feature are very limited, and introduce noise in the model. The plot of number of actors per 5-year career span and the plot of number of actors per movie total are heavily skewed to the right. To reduce the noise, the model will be trained on actors with 3 or more movies
- Initially I trained the model based on the movies in the last 5 years, due to fact that the actor scores are based on the movies that came out up until today, meaning an actors score may be inflated when used in prediction of the rating of the movie which came out before (e.g. 10 years ago)
- In the dataset of () there were 17199 data points with missing rating std dev, upon further investigation we discovered any actor with a single rated movie, would defalt to NULL, as more than 1 data point is required, with that in mind missinf data points will be impued with the average std_dev from actros wtih career span 05
- In the dataset there were 142180 data points with the missing_recent_avg_rating, which will be impuned with the average recent_avg_rating for all actors

# CREATE A MOVIE-LEVEL DATASET FOR 2020-2025
## Load the filtered actor scores into a temporary table
```sql
CREATE TEMPORARY TABLE actor_scores_filtered_imputed AS
SELECT *
FROM read_csv('C:/Users/sasaa/OneDrive/Documents/GOLANG/src/MyVault/NOTES/UC-Davis/S25/ECS171/PROJECT/data/actor_scores_filtered_imputed.csv', delim=',', header=True);
```

## Create a movie-level dataset for movies from 2020–2025
```sql
CREATE TABLE IF NOT EXISTS movie_actor_features_2020_2025_imputed AS
SELECT 
    tp.tconst AS tconst,
    AVG(asf.avg_movie_rating) AS cast_avg_movie_rating,
    AVG(asf.avg_votes_per_movie) AS cast_avg_votes_per_movie,
    AVG(asf.total_movies) AS cast_avg_total_movies,
    AVG(asf.rating_std_dev) AS cast_avg_rating_std_dev,
    AVG(asf.high_rated_movie_count) AS cast_avg_high_rated_movie_count,
    AVG(asf.career_span) AS cast_avg_career_span,
    AVG(asf.recent_avg_rating) AS cast_avg_recent_avg_rating,
    tr.averageRating AS movie_rating
FROM title_principals tp
JOIN title_basics tb ON tp.tconst = tb.tconst
JOIN title_ratings tr ON tb.tconst = tr.tconst
JOIN actor_scores_filtered asf ON tp.nconst = asf.actor
WHERE (tp.category = 'actor' OR tp.category = 'actress')
  AND tb.titleType = 'movie'
  AND tb.startYear >= 2020
  AND tb.startYear <= 2025
GROUP BY tp.tconst, tr.averageRating;
```

## Export the movie-level dataset
```sql
COPY movie_actor_features_2020_2025_imputed TO 'C:/Users/sasaa/OneDrive/Documents/GOLANG/src/MyVault/NOTES/UC-Davis/S25/ECS171/PROJECT/movie_actor_features_2020_2025_imputed.csv' WITH (HEADER, DELIMITER ',');
```

# CORRELATION MATRIX FOR MOVIE ACTOR FEATURES
```python
correlation_matrix = df_movies[['cast_avg_movie_rating', 'cast_avg_votes_per_movie', 'cast_avg_total_movies',
                               'cast_avg_rating_std_dev', 'cast_avg_high_rated_movie_count', 'cast_avg_career_span',
                               'cast_avg_recent_avg_rating']].corr()
print("\nCorrelation Matrix:")
print(correlation_matrix)

# Visualize the correlation matrix
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
plt.title('Correlation Matrix of Movie-Level Features (2020–2025)')
plt.show()
```

# ORDERING DISTRIBUTION
- By inspecting the ordering distribution off the actors, we confirm that the higher order correspond to main actors, whicle lower ordering corresponds to supporting carst.

# TOP 5 TO 7 ACTORS
```sql
CREATE TABLE IF NOT EXISTS movie_actor_features_2020_2025_top_5_7 AS
SELECT 
    tp.tconst AS tconst,
    AVG(asf.avg_movie_rating) AS cast_avg_movie_rating,
    AVG(asf.avg_votes_per_movie) AS cast_avg_votes_per_movie,
    AVG(asf.total_movies) AS cast_avg_total_movies,
    AVG(asf.rating_std_dev) AS cast_avg_rating_std_dev,
    AVG(asf.high_rated_movie_count) AS cast_avg_high_rated_movie_count,
    AVG(asf.career_span) AS cast_avg_career_span,
    AVG(asf.recent_avg_rating) AS cast_avg_recent_avg_rating,
    tr.averageRating AS movie_rating
FROM title_principals tp
JOIN title_basics tb ON tp.tconst = tb.tconst
JOIN title_ratings tr ON tb.tconst = tr.tconst
JOIN actor_scores asf ON tp.nconst = asf.actor
WHERE (tp.category = 'actor' OR tp.category = 'actress')
  AND tb.titleType = 'movie'
  AND tb.startYear >= 2020
  AND tb.startYear <= 2025
  AND tp.ordering BETWEEN 1 AND 7  -- Focus on top 5–7 actors
GROUP BY tp.tconst, tr.averageRating;
```

## Export the updated dataset
```sql
COPY movie_actor_features_2020_2025_top_5_7 TO 'C:/Users/sasaa/OneDrive/Documents/GOLANG/src/MyVault/NOTES/UC-Davis/S25/ECS171/PROJECT/movie_actor_features_2020_2025_top_5_7.csv' WITH (HEADER, DELIMITER ',');
```

# APPROACH 1: ALL ACTORS EQUAL (7 TOTAL)
- Use Ridge Regression model to predict the movies score based on the cast features (cast-avg_movie_rating) which correspond to the features of actor_scored
- once the weights are computed, calculate the actor_score using the weights of the movies model, a
- MSE: 0.6638
- RMSE: 0.8148
- R^2 Score: 0.7283





