
## CREATE TABLE WITH MOVIE WRITER PAIRS
```sql
CREATE TABLE movie_writers_temp AS
SELECT tconst, writer_id
FROM (
    SELECT m.tconst, 
           split_part(tc.writers, ',', 1) AS writer_id,
           1 AS writer_rank
    FROM recent_movies m
    JOIN title_crew tc ON m.tconst = tc.tconst
    WHERE tc.writers IS NOT NULL AND tc.writers != '\N'
      AND split_part(tc.writers, ',', 1) != ''
    UNION ALL
    SELECT m.tconst, 
           split_part(tc.writers, ',', 2) AS writer_id,
           2 AS writer_rank
    FROM recent_movies m
    JOIN title_crew tc ON m.tconst = tc.tconst
    WHERE tc.writers IS NOT NULL AND tc.writers != '\N'
      AND split_part(tc.writers, ',', 2) != ''
    UNION ALL
    SELECT m.tconst, 
           split_part(tc.writers, ',', 3) AS writer_id,
           3 AS writer_rank
    FROM recent_movies m
    JOIN title_crew tc ON m.tconst = tc.tconst
    WHERE tc.writers IS NOT NULL AND tc.writers != '\N'
      AND split_part(tc.writers, ',', 3) != ''
) mw
WHERE writer_id IS NOT NULL
ORDER BY tconst, writer_rank; 
```
- 72974 count

## CREATE DATASET WITH WRITER FEATURES
```sql
CREATE TABLE writer_scores_temp AS
SELECT 
    mw.writer_id,
    COUNT(DISTINCT mw.tconst) AS writer_experience,
    AVG(rm.averageRating) AS writer_quality
FROM movie_writers_temp mw
JOIN recent_movies rm ON mw.tconst = rm.tconst
GROUP BY mw.writer_id
HAVING COUNT(DISTINCT mw.tconst) >= 1;
```
- count 55138
- writer regression data (for generating a single writer score)
## CREATE WRITER REGRESSION DATASET
```sql
CREATE TABLE writer_regression_data AS
SELECT 
    mw.tconst,
    mw.writer_id,
    ws.writer_experience,
    ws.writer_quality,
    mg.averageRating AS target_rating
FROM movie_writers_temp mw
JOIN writer_scores_temp ws ON mw.writer_id = ws.writer_id
JOIN movies_with_genres mg ON mw.tconst = mg.tconst; 
```

## EXPORT WRITERS REGRESSION DATA (NOT USED)
```sql
COPY writer_regression_data TO 'C:/Users/sasaa/OneDrive/Documents/GOLANG/src/MyVault/NOTES/UC-Davis/S25/ECS171/project_imdb_rating/Group_of_machine_learning/actors/data/writer_regression_data.csv' (HEADER, DELIMITER ',');
```

## PERFORMANCE WITH 2 FEATURES

Feature weights:
writer_experience: -0.0012
writer_quality: 1.4457
R^2 score: 0.8656
MSE: 0.3223
RMSE: 0.5677 

## MOVIES WITH WRITER QUALITY FOR 3 MAIN WRITERS
```sql
CREATE TABLE movies_with_movie_writer_quality AS
SELECT 
    m.tconst, m.primaryTitle, m.startYear, m.averageRating, m.numVotes,
    COALESCE(AVG(ws.writer_quality), 5.0) AS movie_writer_quality
FROM movies_with_genres m
LEFT JOIN movie_writers_temp mw ON m.tconst = mw.tconst
LEFT JOIN writer_scores_temp ws ON mw.writer_id = ws.writer_id
GROUP BY m.tconst, m.primaryTitle, m.startYear, m.averageRating, m.numVotes;
```