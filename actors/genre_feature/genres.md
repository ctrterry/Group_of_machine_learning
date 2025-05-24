```sql
CREATE TABLE movies AS
SELECT tb.tconst, tb.primaryTitle, tb.startYear, tb.genres, tr.averageRating, tr.numVotes
FROM title_basics tb
JOIN title_ratings tr ON tb.tconst = tr.tconst
WHERE tb.titleType = 'movie' AND tb.startYear IS NOT NULL AND tb.genres IS NOT NULL;
```
### Convert comma separated genres into an array
```sql
ALTER TABLE movies ADD COLUMN genres_array TEXT[] IF NOT EXISTS;
UPDATE movies SET genres_array = string_split(genres, ',');
```
# DELETE
```sql
DROP TABLE IF EXISTS unique_genres;
CREATE TABLE unique_genres AS
SELECT DISTINCT genre
FROM movies, UNNEST(genres_array) AS genre
WHERE genre IS NOT NULL
ORDER BY genre;
```

# DELETE
```sql
CREATE TABLE unique_genres_mapped AS
SELECT genre,
       CASE 
           WHEN genre = 'Sci-Fi' THEN 'Sci_Fi'
           WHEN genre = 'Film-Noir' THEN 'Film_Noir'
           WHEN genre = 'Reality-TV' THEN 'Reality_TV'
           WHEN genre = 'Game-Show' THEN 'Game_Show'
           ELSE genre
       END AS genre_clean
FROM unique_genres;
```

### Manual One-Hot-Encoding For Genres
```sql
CREATE TABLE movies_with_genres AS
SELECT m.*,
       CASE WHEN array_contains(m.genres_array, 'Action') THEN 1 ELSE 0 END AS genre_Action,
       CASE WHEN array_contains(m.genres_array, 'Adult') THEN 1 ELSE 0 END AS genre_Adult,
       CASE WHEN array_contains(m.genres_array, 'Adventure') THEN 1 ELSE 0 END AS genre_Adventure,
       CASE WHEN array_contains(m.genres_array, 'Animation') THEN 1 ELSE 0 END AS genre_Animation,
       CASE WHEN array_contains(m.genres_array, 'Biography') THEN 1 ELSE 0 END AS genre_Biography,
       CASE WHEN array_contains(m.genres_array, 'Comedy') THEN 1 ELSE 0 END AS genre_Comedy,
       CASE WHEN array_contains(m.genres_array, 'Crime') THEN 1 ELSE 0 END AS genre_Crime,
       CASE WHEN array_contains(m.genres_array, 'Documentary') THEN 1 ELSE 0 END AS genre_Documentary,
       CASE WHEN array_contains(m.genres_array, 'Drama') THEN 1 ELSE 0 END AS genre_Drama,
       CASE WHEN array_contains(m.genres_array, 'Family') THEN 1 ELSE 0 END AS genre_Family,
       CASE WHEN array_contains(m.genres_array, 'Fantasy') THEN 1 ELSE 0 END AS genre_Fantasy,
       CASE WHEN array_contains(m.genres_array, 'Film-Noir') THEN 1 ELSE 0 END AS genre_Film_Noir,
       CASE WHEN array_contains(m.genres_array, 'Game-Show') THEN 1 ELSE 0 END AS genre_Game_Show,
       CASE WHEN array_contains(m.genres_array, 'History') THEN 1 ELSE 0 END AS genre_History,
       CASE WHEN array_contains(m.genres_array, 'Horror') THEN 1 ELSE 0 END AS genre_Horror,
       CASE WHEN array_contains(m.genres_array, 'Music') THEN 1 ELSE 0 END AS genre_Music,
       CASE WHEN array_contains(m.genres_array, 'Musical') THEN 1 ELSE 0 END AS genre_Musical,
       CASE WHEN array_contains(m.genres_array, 'Mystery') THEN 1 ELSE 0 END AS genre_Mystery,
       CASE WHEN array_contains(m.genres_array, 'News') THEN 1 ELSE 0 END AS genre_News,
       CASE WHEN array_contains(m.genres_array, 'Reality-TV') THEN 1 ELSE 0 END AS genre_Reality_TV,
       CASE WHEN array_contains(m.genres_array, 'Romance') THEN 1 ELSE 0 END AS genre_Romance,
       CASE WHEN array_contains(m.genres_array, 'Sci-Fi') THEN 1 ELSE 0 END AS genre_Sci_Fi,
       CASE WHEN array_contains(m.genres_array, 'Sport') THEN 1 ELSE 0 END AS genre_Sport,
       CASE WHEN array_contains(m.genres_array, 'Talk-Show') THEN 1 ELSE 0 END AS genre_Talk_Show,
       CASE WHEN array_contains(m.genres_array, 'Thriller') THEN 1 ELSE 0 END AS genre_Thriller,
       CASE WHEN array_contains(m.genres_array, 'War') THEN 1 ELSE 0 END AS genre_War,
       CASE WHEN array_contains(m.genres_array, 'Western') THEN 1 ELSE 0 END AS genre_Western
FROM movies m;
```
- movie count: 318425, removed those with missing genres, or missing release year
- Almost all movies miss main language, would require to identify the language based on the title, which can be problematic

## EXPORT GENRE SCORES FOR FLASK
```sql
COPY writer_scores_temp TO 'C:/Users/sasaa/OneDrive/Documents/GOLANG/src/MyVault/NOTES/UC-Davis/S25/ECS171/project_imdb_rating/Group_of_machine_learning/actors/writer_quality.csv' (HEADER, DELIMITER ','); 
```