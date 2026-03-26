import re
import pandas as pd

INPUT_PATH = "data/sentiment140_dataset.csv"
OUTPUT_PATH = "data/cleaned_sentiment140.csv"

"""
Function to clean the tweet text 
"""
def clean_text(text):
    if pd.isna(text):
        return ""
    
    text = str(text)                                    # convert to string
    text = text.lower()                                 # lowercase
    text = re.sub(r'https?://\S+|www\.\S+', '', text)   # remove URLs
    text = re.sub(r'@\w+', '', text)                    # remove usernames
    text = re.sub(r'#', '', text)                       # remove hashtag symbol
    text = re.sub(r'\brt\b', '', text)                  # remove rt/RT (repost)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)            # remove special characters, keep numbers
    text = re.sub(r'\s+', ' ', text).strip()            # remove extra spaces

    return text


"""
Function to map the sentiment 
"""
def map_sentiment(label):
    # Sentiment140 uses 0->negative & 4->positve
    if label == 0:
        return "Negative"
    elif label == 4:
        return "Positive"
    return None

"""
Main Function
"""
def main():
    print("Reading original dataset from: ", INPUT_PATH)

    # Original Sentiment140 format
    org_df = pd.read_csv(
        INPUT_PATH, 
        encoding='latin-1', 
        header=None,
        names=["target", "ids", "date", "flag", "user", "text"]
    )

    # Take only target & text columns
    new_df = org_df[["target", "text"]].copy()
    
    # Map target labels
    new_df["sentiment_label"] = new_df["target"].apply(map_sentiment)

    # Clean text
    new_df["clean_text"] = new_df["text"].apply(clean_text)

    # Keep only sentiment_label & clean_text columns
    new_df = new_df[["sentiment_label", "clean_text"]]
    
    # Remove rows with missing values
    new_df = new_df.dropna()        

    # Remove rows with empty cleaned texts 
    new_df = new_df[new_df["clean_text"] != ""]

    # Save cleaned datset
    new_df.to_csv(OUTPUT_PATH, index=False)

    print("Cleaned dataset saved to: ", OUTPUT_PATH)
    print(new_df.head(5))
    print(new_df.shape)


if __name__ == "__main__":
    main()