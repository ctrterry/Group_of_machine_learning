

# CREATE TABLES FOR RAW IMDB DATA
```
.open C:/Users/sasaa/OneDrive/Documents/GOLANG/src/MyVault/NOTES/UC-Davis/S25/ECS171/project_imdb_rating/Group_of_machine_learning/Sasha_Directory/actors/db/imdb.duckdb
```

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

## CREATE TABLE WITH MOVIES IN 2020-2025
```sql
CREATE TABLE recent_movies AS
SELECT tb.tconst, tb.primaryTitle, tb.startYear, tr.averageRating, tr.numVotes
FROM title_basics tb
JOIN title_ratings tr ON tb.tconst = tr.tconst
WHERE tb.titleType = 'movie'
  AND tb.startYear IS NOT NULL
  AND CAST(tb.startYear AS INTEGER) BETWEEN 2020 AND 2025;
  ```
- 53005 movies

# MOVIES FEATURES TABLE
```sql
CREATE TABLE movies_features_DELETE AS
SELECT 
    rm.tconst, rm.primaryTitle, rm.startYear, rm.averageRating, rm.numVotes,
    gs.genre_score,
    mw.movie_writer_quality,
    ma.movie_actor_score,
    mb.budget,
    md.director_quality
FROM recent_movies rm
INNER JOIN movies_with_genre_score gs ON rm.tconst = gs.tconst
INNER JOIN movies_with_movie_writer_quality mw ON rm.tconst = mw.tconst
INNER JOIN movies_with_actor_scores ma ON rm.tconst = ma.tconst
INNER JOIN movies_with_budgets mb ON rm.tconst = mb.tconst
INNER JOIN movies_with_director_quality md ON rm.tconst = md.tconst;
```
## IMPUTATION
```sql
SELECT
    COUNT(*) AS total_movies,
    SUM(CASE WHEN md.tconst IS NOT NULL THEN 1 ELSE 0 END) AS matching_director_quality,
    SUM(CASE WHEN md.tconst IS NULL THEN 1 ELSE 0 END) AS missing_director_quality
FROM recent_movies rm
LEFT JOIN movies_with_director_quality md ON rm.tconst = md.tconst;
```

missing genre 960, missing writer quality 940, missing movie actor score 9895, missing budget 940+24, missing director quality 793

## EXPORT DROPPED MOVIES

```sql
COPY (
    SELECT
    rm.tconst, rm.primaryTitle, rm.startYear,
    CASE WHEN gs.tconst IS NULL THEN 'Missing' ELSE 'Present' END AS genre_score_status,
    CASE WHEN mw.tconst IS NULL THEN 'Missing' ELSE 'Present' END AS movie_writer_quality_status,
    CASE WHEN ma.tconst IS NULL THEN 'Missing' ELSE 'Present' END AS movie_actor_score_status,
    CASE WHEN mb.tconst IS NULL THEN 'Missing' ELSE 'Present' END AS budget_status,
    CASE WHEN md.tconst IS NULL THEN 'Missing' ELSE 'Present' END AS director_quality_status
FROM recent_movies rm
LEFT JOIN movies_with_genre_score gs ON rm.tconst = gs.tconst
LEFT JOIN movies_with_movie_writer_quality mw ON rm.tconst = mw.tconst
LEFT JOIN movies_with_actor_scores ma ON rm.tconst = ma.tconst
LEFT JOIN movies_with_budgets mb ON rm.tconst = mb.tconst
LEFT JOIN movies_with_director_quality md ON rm.tconst = md.tconst
WHERE gs.tconst IS NULL
   OR mw.tconst IS NULL
   OR ma.tconst IS NULL
   OR mb.tconst IS NULL
   OR md.tconst IS NULL;
) TO 'dropped_movies.csv' WITH (HEADER 1, DELIMITER ',');
```
## NUMBER OF DATA POINTS WITH MISSING FEATURES
```sql
SELECT
    SUM(CASE WHEN gs.tconst IS NULL THEN 1 ELSE 0 END) AS missing_genre_score,
    SUM(CASE WHEN mw.tconst IS NULL THEN 1 ELSE 0 END) AS missing_movie_writer_quality,
    SUM(CASE WHEN ma.tconst IS NULL THEN 1 ELSE 0 END) AS missing_movie_actor_score,
    SUM(CASE WHEN mb.tconst IS NULL THEN 1 ELSE 0 END) AS missing_budget,
    SUM(CASE WHEN md.tconst IS NULL THEN 1 ELSE 0 END) AS missing_director_quality,
    SUM(CASE WHEN gs.tconst IS NULL OR mw.tconst IS NULL OR ma.tconst IS NULL OR mb.tconst IS NULL OR md.tconst IS NULL THEN 1 ELSE 0 END) AS missing_any,
    SUM(CASE WHEN gs.tconst IS NULL AND ma.tconst IS NULL THEN 1 ELSE 0 END) AS missing_genre_and_actor,
    SUM(CASE WHEN mb.tconst IS NULL AND gs.tconst IS NULL THEN 1 ELSE 0 END) AS missing_budget_and_genre
FROM recent_movies rm
LEFT JOIN movies_with_genre_score gs ON rm.tconst = gs.tconst
LEFT JOIN movies_with_movie_writer_quality mw ON rm.tconst = mw.tconst
LEFT JOIN movies_with_actor_scores ma ON rm.tconst = ma.tconst
LEFT JOIN movies_with_budgets mb ON rm.tconst = mb.tconst
LEFT JOIN movies_with_director_quality md ON rm.tconst = md.tconst;
```