

# CREATE TABLES FOR RAW IMDB DATA
```
.open C:/Users/sasaa/OneDrive/Documents/GOLANG/src/MyVault/NOTES/UC-Davis/S25/ECS171/project_imdb_rating/Group_of_machine_learning/actors/db/imdb.duckdb
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

## CREATE TABLE WITH MOVIES IN 2020-2005
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