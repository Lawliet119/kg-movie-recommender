/**
 * Movie Knowledge Graph Data — MetaQA format: subject|relation|object
 */
export const RAW_TRIPLES = [
  // Christopher Nolan
  "Inception|directed_by|Christopher Nolan","Inception|starred_actors|Leonardo DiCaprio","Inception|starred_actors|Tom Hardy","Inception|starred_actors|Cillian Murphy","Inception|has_genre|Sci-Fi","Inception|has_genre|Action","Inception|has_genre|Thriller","Inception|release_year|2010",
  "The Dark Knight|directed_by|Christopher Nolan","The Dark Knight|starred_actors|Christian Bale","The Dark Knight|starred_actors|Morgan Freeman","The Dark Knight|starred_actors|Gary Oldman","The Dark Knight|has_genre|Action","The Dark Knight|has_genre|Crime","The Dark Knight|has_genre|Thriller","The Dark Knight|release_year|2008",
  "Interstellar|directed_by|Christopher Nolan","Interstellar|starred_actors|Matthew McConaughey","Interstellar|starred_actors|Anne Hathaway","Interstellar|has_genre|Sci-Fi","Interstellar|has_genre|Adventure","Interstellar|has_genre|Drama","Interstellar|release_year|2014",
  "Oppenheimer|directed_by|Christopher Nolan","Oppenheimer|starred_actors|Cillian Murphy","Oppenheimer|starred_actors|Robert Downey Jr.","Oppenheimer|starred_actors|Matt Damon","Oppenheimer|has_genre|Drama","Oppenheimer|has_genre|Thriller","Oppenheimer|release_year|2023",
  "Tenet|directed_by|Christopher Nolan","Tenet|has_genre|Action","Tenet|has_genre|Sci-Fi","Tenet|has_genre|Thriller","Tenet|release_year|2020",
  "Memento|directed_by|Christopher Nolan","Memento|starred_actors|Guy Pearce","Memento|has_genre|Thriller","Memento|has_genre|Mystery","Memento|release_year|2000",
  // Quentin Tarantino
  "Pulp Fiction|directed_by|Quentin Tarantino","Pulp Fiction|starred_actors|Samuel L. Jackson","Pulp Fiction|starred_actors|John Travolta","Pulp Fiction|starred_actors|Uma Thurman","Pulp Fiction|has_genre|Crime","Pulp Fiction|has_genre|Drama","Pulp Fiction|release_year|1994",
  "Django Unchained|directed_by|Quentin Tarantino","Django Unchained|starred_actors|Leonardo DiCaprio","Django Unchained|starred_actors|Samuel L. Jackson","Django Unchained|starred_actors|Jamie Foxx","Django Unchained|has_genre|Drama","Django Unchained|has_genre|Western","Django Unchained|release_year|2012",
  "Inglourious Basterds|directed_by|Quentin Tarantino","Inglourious Basterds|starred_actors|Brad Pitt","Inglourious Basterds|starred_actors|Christoph Waltz","Inglourious Basterds|has_genre|Adventure","Inglourious Basterds|has_genre|Drama","Inglourious Basterds|has_genre|War","Inglourious Basterds|release_year|2009",
  "Kill Bill: Volume 1|directed_by|Quentin Tarantino","Kill Bill: Volume 1|starred_actors|Uma Thurman","Kill Bill: Volume 1|has_genre|Action","Kill Bill: Volume 1|has_genre|Crime","Kill Bill: Volume 1|has_genre|Thriller","Kill Bill: Volume 1|release_year|2003",
  "Once Upon a Time in Hollywood|directed_by|Quentin Tarantino","Once Upon a Time in Hollywood|starred_actors|Leonardo DiCaprio","Once Upon a Time in Hollywood|starred_actors|Brad Pitt","Once Upon a Time in Hollywood|starred_actors|Margot Robbie","Once Upon a Time in Hollywood|has_genre|Comedy","Once Upon a Time in Hollywood|has_genre|Drama","Once Upon a Time in Hollywood|release_year|2019",
  // Martin Scorsese
  "The Wolf of Wall Street|directed_by|Martin Scorsese","The Wolf of Wall Street|starred_actors|Leonardo DiCaprio","The Wolf of Wall Street|starred_actors|Margot Robbie","The Wolf of Wall Street|has_genre|Crime","The Wolf of Wall Street|has_genre|Comedy","The Wolf of Wall Street|has_genre|Drama","The Wolf of Wall Street|release_year|2013",
  "Goodfellas|directed_by|Martin Scorsese","Goodfellas|starred_actors|Robert De Niro","Goodfellas|starred_actors|Joe Pesci","Goodfellas|has_genre|Crime","Goodfellas|has_genre|Drama","Goodfellas|release_year|1990",
  "Shutter Island|directed_by|Martin Scorsese","Shutter Island|starred_actors|Leonardo DiCaprio","Shutter Island|starred_actors|Mark Ruffalo","Shutter Island|has_genre|Thriller","Shutter Island|has_genre|Mystery","Shutter Island|release_year|2010",
  "The Irishman|directed_by|Martin Scorsese","The Irishman|starred_actors|Robert De Niro","The Irishman|starred_actors|Al Pacino","The Irishman|starred_actors|Joe Pesci","The Irishman|has_genre|Crime","The Irishman|has_genre|Drama","The Irishman|release_year|2019",
  "The Departed|directed_by|Martin Scorsese","The Departed|starred_actors|Leonardo DiCaprio","The Departed|starred_actors|Matt Damon","The Departed|starred_actors|Mark Ruffalo","The Departed|has_genre|Crime","The Departed|has_genre|Thriller","The Departed|has_genre|Drama","The Departed|release_year|2006",
  // Steven Spielberg
  "Saving Private Ryan|directed_by|Steven Spielberg","Saving Private Ryan|starred_actors|Tom Hanks","Saving Private Ryan|starred_actors|Matt Damon","Saving Private Ryan|has_genre|Drama","Saving Private Ryan|has_genre|War","Saving Private Ryan|release_year|1998",
  "Jurassic Park|directed_by|Steven Spielberg","Jurassic Park|starred_actors|Samuel L. Jackson","Jurassic Park|starred_actors|Jeff Goldblum","Jurassic Park|has_genre|Sci-Fi","Jurassic Park|has_genre|Adventure","Jurassic Park|has_genre|Action","Jurassic Park|release_year|1993",
  "Catch Me If You Can|directed_by|Steven Spielberg","Catch Me If You Can|starred_actors|Leonardo DiCaprio","Catch Me If You Can|starred_actors|Tom Hanks","Catch Me If You Can|has_genre|Crime","Catch Me If You Can|has_genre|Comedy","Catch Me If You Can|has_genre|Drama","Catch Me If You Can|release_year|2002",
  "Schindler's List|directed_by|Steven Spielberg","Schindler's List|starred_actors|Liam Neeson","Schindler's List|has_genre|Drama","Schindler's List|has_genre|War","Schindler's List|release_year|1993",
  // David Fincher
  "Fight Club|directed_by|David Fincher","Fight Club|starred_actors|Brad Pitt","Fight Club|starred_actors|Edward Norton","Fight Club|has_genre|Drama","Fight Club|has_genre|Thriller","Fight Club|release_year|1999",
  "Se7en|directed_by|David Fincher","Se7en|starred_actors|Brad Pitt","Se7en|starred_actors|Morgan Freeman","Se7en|has_genre|Crime","Se7en|has_genre|Thriller","Se7en|has_genre|Mystery","Se7en|release_year|1995",
  "Gone Girl|directed_by|David Fincher","Gone Girl|starred_actors|Ben Affleck","Gone Girl|has_genre|Thriller","Gone Girl|has_genre|Drama","Gone Girl|has_genre|Mystery","Gone Girl|release_year|2014",
  "The Social Network|directed_by|David Fincher","The Social Network|starred_actors|Jesse Eisenberg","The Social Network|has_genre|Drama","The Social Network|release_year|2010",
  // Denis Villeneuve
  "Dune|directed_by|Denis Villeneuve","Dune|starred_actors|Timothée Chalamet","Dune|starred_actors|Zendaya","Dune|has_genre|Sci-Fi","Dune|has_genre|Adventure","Dune|has_genre|Drama","Dune|release_year|2021",
  "Blade Runner 2049|directed_by|Denis Villeneuve","Blade Runner 2049|starred_actors|Ryan Gosling","Blade Runner 2049|has_genre|Sci-Fi","Blade Runner 2049|has_genre|Thriller","Blade Runner 2049|release_year|2017",
  "Arrival|directed_by|Denis Villeneuve","Arrival|starred_actors|Amy Adams","Arrival|has_genre|Sci-Fi","Arrival|has_genre|Drama","Arrival|release_year|2016",
  // Other Iconic Films
  "The Matrix|directed_by|The Wachowskis","The Matrix|starred_actors|Keanu Reeves","The Matrix|has_genre|Sci-Fi","The Matrix|has_genre|Action","The Matrix|release_year|1999",
  "John Wick|directed_by|Chad Stahelski","John Wick|starred_actors|Keanu Reeves","John Wick|has_genre|Action","John Wick|has_genre|Thriller","John Wick|release_year|2014",
  "The Shawshank Redemption|directed_by|Frank Darabont","The Shawshank Redemption|starred_actors|Morgan Freeman","The Shawshank Redemption|starred_actors|Tim Robbins","The Shawshank Redemption|has_genre|Drama","The Shawshank Redemption|release_year|1994",
  "Forrest Gump|directed_by|Robert Zemeckis","Forrest Gump|starred_actors|Tom Hanks","Forrest Gump|has_genre|Drama","Forrest Gump|has_genre|Romance","Forrest Gump|release_year|1994",
  "The Godfather|directed_by|Francis Ford Coppola","The Godfather|starred_actors|Al Pacino","The Godfather|starred_actors|Marlon Brando","The Godfather|has_genre|Crime","The Godfather|has_genre|Drama","The Godfather|release_year|1972",
  "Parasite|directed_by|Bong Joon-ho","Parasite|starred_actors|Song Kang-ho","Parasite|has_genre|Thriller","Parasite|has_genre|Drama","Parasite|has_genre|Comedy","Parasite|release_year|2019",
  "Titanic|directed_by|James Cameron","Titanic|starred_actors|Leonardo DiCaprio","Titanic|starred_actors|Kate Winslet","Titanic|has_genre|Drama","Titanic|has_genre|Romance","Titanic|release_year|1997",
  "Avatar|directed_by|James Cameron","Avatar|has_genre|Sci-Fi","Avatar|has_genre|Action","Avatar|has_genre|Adventure","Avatar|release_year|2009",
  "Gladiator|directed_by|Ridley Scott","Gladiator|starred_actors|Russell Crowe","Gladiator|has_genre|Action","Gladiator|has_genre|Drama","Gladiator|has_genre|Adventure","Gladiator|release_year|2000",
  "The Revenant|directed_by|Alejandro Iñárritu","The Revenant|starred_actors|Leonardo DiCaprio","The Revenant|starred_actors|Tom Hardy","The Revenant|has_genre|Adventure","The Revenant|has_genre|Drama","The Revenant|release_year|2015",
  "The Lord of the Rings|directed_by|Peter Jackson","The Lord of the Rings|starred_actors|Elijah Wood","The Lord of the Rings|starred_actors|Ian McKellen","The Lord of the Rings|has_genre|Fantasy","The Lord of the Rings|has_genre|Adventure","The Lord of the Rings|has_genre|Action","The Lord of the Rings|release_year|2001",
  "The Avengers|directed_by|Joss Whedon","The Avengers|starred_actors|Robert Downey Jr.","The Avengers|starred_actors|Scarlett Johansson","The Avengers|starred_actors|Samuel L. Jackson","The Avengers|starred_actors|Mark Ruffalo","The Avengers|has_genre|Action","The Avengers|has_genre|Sci-Fi","The Avengers|has_genre|Adventure","The Avengers|release_year|2012",
  "Iron Man|directed_by|Jon Favreau","Iron Man|starred_actors|Robert Downey Jr.","Iron Man|has_genre|Action","Iron Man|has_genre|Sci-Fi","Iron Man|has_genre|Adventure","Iron Man|release_year|2008",
  "Mad Max: Fury Road|directed_by|George Miller","Mad Max: Fury Road|starred_actors|Tom Hardy","Mad Max: Fury Road|starred_actors|Charlize Theron","Mad Max: Fury Road|has_genre|Action","Mad Max: Fury Road|has_genre|Sci-Fi","Mad Max: Fury Road|has_genre|Adventure","Mad Max: Fury Road|release_year|2015",
];

/** Parse raw triples into structured KG */
export function buildKnowledgeGraph(rawTriples = RAW_TRIPLES) {
  const entities = {};
  const triples = [];
  const adjacency = {};
  const reverseAdjacency = {};
  const typeMap = {
    directed_by: { head: 'movie', tail: 'person' },
    starred_actors: { head: 'movie', tail: 'person' },
    written_by: { head: 'movie', tail: 'person' },
    has_genre: { head: 'movie', tail: 'genre' },
    release_year: { head: 'movie', tail: 'year' },
    has_imdb_rating: { head: 'movie', tail: 'rating' },
    has_tags: { head: 'movie', tail: 'tag' },
    in_language: { head: 'movie', tail: 'language' },
  };

  for (const line of rawTriples) {
    const parts = line.split('|');
    if (parts.length !== 3) continue;
    const [subject, relation, object] = parts.map(s => s.trim());
    const ti = typeMap[relation] || { head: 'unknown', tail: 'unknown' };
    const headId = `${ti.head}:${subject.toLowerCase().replace(/[^a-z0-9]+/g, '_')}`;
    const tailId = `${ti.tail}:${object.toLowerCase().replace(/[^a-z0-9]+/g, '_')}`;

    if (!entities[headId]) entities[headId] = { id: headId, name: subject, type: ti.head };
    if (!entities[tailId]) entities[tailId] = { id: tailId, name: object, type: ti.tail };

    triples.push({ head: headId, relation, tail: tailId });
    if (!adjacency[headId]) adjacency[headId] = [];
    adjacency[headId].push({ relation, target: tailId });
    if (!reverseAdjacency[tailId]) reverseAdjacency[tailId] = [];
    reverseAdjacency[tailId].push({ relation, target: headId });
  }

  const movieCount = Object.values(entities).filter(e => e.type === 'movie').length;
  return {
    entities, triples, adjacency, reverseAdjacency,
    stats: {
      totalEntities: Object.keys(entities).length,
      totalTriples: triples.length,
      movieCount,
      personCount: Object.values(entities).filter(e => e.type === 'person').length,
      genreCount: Object.values(entities).filter(e => e.type === 'genre').length,
    },
  };
}
