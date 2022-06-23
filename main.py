import cv2
import os
import math
import requests
import numpy as np
import json
import face_recognition
from sklearn.metrics.pairwise import cosine_similarity
from requests_oauthlib import OAuth1
from multiprocessing import Pool, cpu_count


def get_tweets_from_screen_name(screen_name, max_tweets=100):
    """
    Retrieve the last n tweets (including retweets) for a given user
    (max n = 3200)
    """
    params = {
        "screen_name": screen_name,
        "count": max_tweets,
        "include_rts": "true",
        "exclude_replies": "false",
        "tweet_mode": "extended"
    }
    # A cursor in instanciated in order to navigate into the tweets
    cursor = None
    tweets=[]
    while len(tweets) < max_tweets:
        if cursor is not None:
            params["max_id"] = cursor
            start_index = 1
        else:
            start_index = 0
        
        # Start index allows to ignore the first result when it has already been extracted (cf param "max_id" in Twitter API)
        r = requests.get("https://api.twitter.com/1.1/statuses/user_timeline.json", params=params, auth=auth, timeout=5)
        if r.status_code != 200:
            print(r.text)
            return tweets

        try:
            results = r.json()[start_index:]
        except:
            return tweets

        if len(results) > 0:
            for result in results:
                tweets.append(result)
                if len(tweets) == max_tweets:
                    return tweets
            cursor = results[-1]["id"]
        else:
            return tweets
    return tweets


def download_image(url):
    """
    Download an image based on its URL and returns the image object
    """
    r = requests.get(url)
    np_img = np.frombuffer(r.content, np.uint8)
    try:
        return cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    except:
        return None

def get_images_from_tweets(tweets):
    """
    Download all images from a list of tweets
    """
    images_urls = []
    for tweet in tweets:
        if "extended_entities" in tweet:
            images_urls += [media["media_url"] for media in tweet["extended_entities"]["media"] if media["type"] == "photo"]
        if "extended_entities" in tweet.get("retweeted_status", {}):
            images_urls += [media["media_url"] for media in tweet["retweeted_status"]["extended_entities"]["media"] if media["type"] == "photo"]
    pool = Pool(cpu_count())
    images = pool.map(download_image, images_urls)
    pool.close()
    pool.join()
    return [im for im in images if im is not None]


def get_faces_from_image(image):
    """
    Retrieve a list of faces from an image by using the OpenCV
    'haarcascade_frontalface_default' model
    """
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=3,
            minSize=(30, 30)
        )
    except:
        faces = []

    for (x, y, w, h) in faces:
        face = image[y:y+h,x:x+w]
        yield face

def get_face_encoding(face):
    """
    Transform the image in RGB and return its embedding
    by using the face_recognition library
    """
    try:
        rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        return face_recognition.face_encodings(rgb)[0]
    except:
        return None


def get_reference_face_encoding(url):
    """
        Create a baseline encoding from the first face retrieved on the
        reference image provided by the user
        (Assumes that there is one single face on the image)
    """
    try:
        image = download_image(url)
    except:
        raise Exception("The reference URL is not valid or not reachable")
    for face in get_faces_from_image(image):
        return get_face_encoding(face)
    raise Exception("The system didn't detect any face from your reference url")   
        

def transform_image_opacity(image, similarity, threshold):
    """
    Decrease the opactiy of the picture if the similarity is below
    a given threshold
    """
    if similarity < threshold:
        alpha_coeff = 100
    else:
        alpha_coeff = 255
        
    b_channel, g_channel, r_channel = cv2.split(image)
    a_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * alpha_coeff
    image = cv2.merge((b_channel, g_channel, r_channel, a_channel))
    return image



def draw_face_mosaic(faces):
    """
    Build a square of faces based on a list of faces
    """
    list2d = []
    chunk_size = int(math.sqrt(len(faces)))
    if chunk_size == 0:
        raise Exception("Not enough faces on the pictures of this account")
    for i in range(0, len(faces), chunk_size):
        list2d.append(faces[i: i + chunk_size])
    return cv2.vconcat([cv2.hconcat(list_h) for list_h in list(list2d)[:-1]])



def get_mosaic_and_ratio(images, reference_encoding, threshold):
    
    all_faces = []
    all_encodings = []
    
    # Get all encodings
    for image in images:
        for face in get_faces_from_image(image):
            face_encoding = get_face_encoding(face)
            if face_encoding is not None:
                all_faces.append(cv2.resize(face, (100, 100), cv2.INTER_NEAREST))
                all_encodings.append(face_encoding)
    
    # Compute the similarity with the reference image
    face_similarities = []
    for face, encoding in zip(all_faces, all_encodings):
        face_similarities.append((face, cosine_similarity([reference_encoding], [encoding])[0][0]))

    # Sort the list of images and transform the opacity based on similarity 
    face_similarities.sort(key=lambda x:x[1], reverse=True)
    sorted_faces = [transform_image_opacity(sim[0], sim[1], threshold) for sim in face_similarities]

    # Compute the ratio : number of faces above threshold divided by number of faces
    similarities_above_threshold = [sim[1] for sim in face_similarities if sim[1] >= threshold]

    if len(face_similarities) > 0:
        ratio = len(similarities_above_threshold) / len(face_similarities)
    else:
        ratio = 0
   
    # Draw the face mosaic
    final_image = draw_face_mosaic(sorted_faces) 
    return final_image, ratio
   

if __name__ == "__main__":

    # Insert here the Twitter credentials
    credentials = json.load(open("credentials.json", "r"))
    auth = OAuth1(credentials["consumer_key"], credentials["consumer_secret"], credentials["access_token"], credentials["access_token_secret"])
   
    # Input data
    twitter_screen_name = input("Please indicate a Twitter screen name to analyse:\n> ")
    baseline_url = input("Please indicate an URL to the reference face:\n> ")
    threshold = float(input("Please insert a similarity threshold (suggested: 0.9):\n> "))
    max_tweets = int(input("How many tweets do you want to analyse (max = 3200)?\n>"))

    # Build baseline embedding
    baseline_embedding = get_reference_face_encoding(baseline_url)
    
    # Get tweets
    tweets = get_tweets_from_screen_name(twitter_screen_name, max_tweets)
    print(f"{len(tweets)} tweets found")

    # Get images
    images = get_images_from_tweets(tweets)
    print(f"{len(images)} images found")

    # Compute the mosaic and ratio
    final_image, ratio = get_mosaic_and_ratio(images, baseline_embedding, threshold)
    
    print("\n-------------------------------\n")

    print(f"Ratio of faces corresponding to the reference face: {round(ratio,4)*100} %")

    cv2.imwrite(f"examples/{twitter_screen_name}.png", final_image)
    print(f"The mosaic has been generated : examples/{twitter_screen_name}.png\n\n")





    
    

