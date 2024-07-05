import pandas as pd

comments_data = pd.read_csv("data/comments.csv")
posts_data = pd.read_csv("data/posts.csv")

bad_posts = [
    "1dtmogb",
    "1aof69c",
    "1bapah6",
    "18i0lqf",
    "116a9qq",
    "146pc14",
    "16pp2h6",
    "1cja071",
    "106yqxk",
    "1cqx1iz",
    "1atdd05",
    "1527ivl",
    "17k8rsn",
    "14dew25",
    "15b34ng",
    "16ofae7",
    "13l893a",
    "13dufss",
    "175s6nt",
    "16xh795",
    "1cksd4x",
    "133gqee",
    "11yiygr",
    "13r3v36",
    "150sshi",
    "159yd7n",
    "1cfw1ra",
    "172dubs",
    "12s31nh",
    "12qhb8x",
    "18501uz",
    "1czg3x6",
    "15rtqgc",
    "164jmfw",
    "13l81jl",
    "1675zhp",
    "13r3v36",
    "1808tpw",
    "18qbdzh",
    "1beimov",
    "112fp9z",
]

# Remove bad posts
comments_data = comments_data[~comments_data["post_id"].isin(bad_posts)]
posts_data = posts_data[~posts_data["post_id"].isin(bad_posts)]

# Save the cleaned data
comments_data.to_csv("data/comments_cleaned.csv", index=False)
posts_data.to_csv("data/posts_cleaned.csv", index=False)
