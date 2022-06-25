import cv2
import os
import math
import requests
import numpy as np
import json
import uuid
from multiprocessing import Pool, cpu_count
import face_recognition
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
from requests_oauthlib import OAuth1

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

def download_video(url):
    """
    Download an image based on its URL and returns the image object
    """
    r = requests.get(url)
    path = os.path.join("_temp", f"{str(uuid.uuid4())}.mp4")
    with open(path, "wb") as out:
        out.write(r.content)
    return path


def get_images_and_videos_from_tweets(tweets):
    """
    Download all images from a list of tweets
    """
    images_urls = []
    videos_urls = []
    for tweet in tweets:

        medias = tweet.get("extended_entities", {}).get("media", [])
        medias += tweet.get("retweeted_status", {}).get("extended_entities", {}).get("media", [])
        for media in medias:
            if media["type"] == "photo":
                images_urls.append(media["media_url"])
            elif media["type"] == "video":
                variants = [variant for variant in media["video_info"]["variants"] if variant["content_type"] == "video/mp4"]
                if len(variants) > 0:
                    videos_urls.append(variants[0]["url"])    
    return images_urls, videos_urls


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

def get_faces_from_video(video_path):
    """
    Retrieve a list of unique faces from a video
    """
    cap = cv2.VideoCapture(video_path)
    i = 0
    # Get the faces every 100 panes of the video
    faces = []
    while(cap.isOpened()):
        response, img = cap.read()
        if response == False:
            break
        if i % 100 == 0:
            faces += get_faces_from_image(img)    
        i += 1
    cap.release()

    # Embed the faces for allowing clustering
    all_encodings = []
    all_faces = []
    for face in faces:
        face_encoding = get_face_encoding(face)
        if face_encoding is not None:
            all_faces.append(face)
            all_encodings.append(face_encoding)
    
    if len(all_faces) == 0:
        return []
    
    # Apply DBSCAN clustering on the faces to cluster unique faces
    clustering = DBSCAN(metric="euclidean", n_jobs=10, min_samples=3)
    clustering.fit(all_encodings)
    
    # Get one single face per cluster
    unique_faces = []
    furnished_cluster_ids = set()
   
    for cluster_id, face in zip(clustering.labels_, all_faces):
        if cluster_id not in furnished_cluster_ids:
            unique_faces.append(face)
            furnished_cluster_ids.add(cluster_id)
    return unique_faces


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
        Create a reference encoding from the first face retrieved on the
        reference image provided by the user
        (Assumes that there is one single face on the image)
    """
    try:
        image = download_image(url)
    except:
        raise Exception("The reference URL is not valid or not reachable")
    for face in get_faces_from_image(image):
        encoding = get_face_encoding(face)
        if encoding is not None:
            return encoding
    raise Exception("The system didn't detect any face from your reference url")   
        

def transform_image_opacity(image, similarity, similarity_threshold):
    """
    Decrease the opactiy of the picture if the similarity is below
    a given threshold
    """
    if similarity < similarity_threshold:
        alpha_coeff = 100
    else:
        alpha_coeff = 255
        
    b_channel, g_channel, r_channel = cv2.split(image)
    a_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * alpha_coeff
    image = cv2.merge((b_channel, g_channel, r_channel, a_channel))
    return image





def get_faces_from_images_urls(images_urls):
    """
    Download images with multiprocessing and get the faces
    from all the images
    """
    pool = Pool(cpu_count())
    images = pool.map(download_image, images_urls)
    pool.close()
    pool.join()
    faces = []
    for image in images:
        faces += get_faces_from_image(image)
    return faces
   

def get_faces_from_videos_urls(videos_urls):
    """
    Get the faces from all the videos
    """
    pool = Pool(cpu_count())
    video_paths = pool.map(download_video, videos_urls)
    pool.close()
    pool.join()
    faces = []
    for i, video_path in enumerate(video_paths):
        faces += get_faces_from_video(video_path)
        os.remove(video_path)
    return faces



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



def get_mosaic_and_stats(faces, reference_encoding, similarity_threshold):
    """
    Build the mosaic of faces by ranking the faces based on their similarity
    with the reference encoding
    """
    all_faces = []
    all_encodings = []
    
    # Get all encodings
    for face in faces:
        face_encoding = get_face_encoding(face)
        if face_encoding is not None:
            all_faces.append(cv2.resize(face, (100, 100), cv2.INTER_NEAREST))
            all_encodings.append(face_encoding)
        
    
    # Compute the similarity with the reference image
    all_similarities = cosine_similarity([reference_encoding], all_encodings)[0]
    face_similarities = list(zip(all_faces, all_similarities))

    # Sort the list of images and transform the opacity based on similarity 
    face_similarities.sort(key=lambda x:x[1], reverse=True)
    sorted_faces = [transform_image_opacity(face, similarity, similarity_threshold) for face, similarity in face_similarities]

    # Compute the ratio : number of faces above threshold divided by number of faces
    similarities_above_threshold = [sim[1] for sim in face_similarities if sim[1] >= similarity_threshold]
   
    # Draw the face mosaic
    final_image = draw_face_mosaic(sorted_faces) 
    return final_image, len(similarities_above_threshold), len(all_faces)

def launch_process(twitter_screen_name, reference_url, similarity_threshold, max_tweets):
    """
    Launch the full process for one account and one reference image
    """

    # Build reference embedding
    reference_embedding = get_reference_face_encoding(reference_url)
    
    # Get tweets
    print(f"\nGetting the latest {max_tweets} tweets from the user...")
    tweets = get_tweets_from_screen_name(twitter_screen_name, max_tweets)
    print(f"{len(tweets)} tweets found\n")

    # Get images
    print("Getting images and videos URLS...")
    images_urls, videos_urls = get_images_and_videos_from_tweets(tweets)
    print(f"{len(images_urls)} images and {len(videos_urls)} videos found\n")

    # Extract faces from images urls
    print("Extracting faces from images...")
    images_faces = get_faces_from_images_urls(images_urls)
    print(f"{len(images_faces)} potential faces found on images\n")
    print("Extracting faces from videos...")
    videos_faces = get_faces_from_videos_urls(videos_urls)
    print(f"{len(videos_faces)} potential faces found on videos\n")

    faces = images_faces + videos_faces

    # Compute the mosaic and ratio
    print("Computing ratios and generating mosaic...")
    final_image, count_reference, count_all_faces = get_mosaic_and_stats(faces, reference_embedding, similarity_threshold)
    
    print("\n-------------------------------\n")

    total_medias = len(images_urls) + len(videos_urls)
    total_faces = len(faces)

    print(f"The reference face appears {count_reference} times out of {len(images_urls) + len(videos_urls)} media shared by {twitter_screen_name} (Ratio: {round(count_reference / total_medias * 100, 2)}%)")
    print(f"The reference face appears {count_reference} times out of {total_faces} valid faces popping up in the media shared by {twitter_screen_name} (Ratio: {round(count_reference / total_faces * 100, 2)}%)")

    cv2.imwrite(f"examples/{twitter_screen_name}.png", final_image)
    print(f"\nThe mosaic has been generated : examples/{twitter_screen_name}.png\n\n")
    return {
        "total_images": len(images_urls),
        "total_videos": len(videos_urls),
        "total_faces": total_faces,
        "count_reference_face": count_reference
    }




   

if __name__ == "__main__":

    # Insert here the Twitter credentials
    credentials = json.load(open("credentials.json", "r"))
    auth = OAuth1(credentials["consumer_key"], credentials["consumer_secret"], credentials["access_token"], credentials["access_token_secret"])
   
    # Input data
    twitter_screen_name = input("Please indicate a Twitter screen name to analyse:\n> ")
    reference_url = input("\nPlease indicate an URL to the reference face:\n> ")
    similarity_threshold = float(input("\nPlease insert a similarity threshold (suggested: 0.9):\n> "))
    max_tweets = int(input("\nHow many tweets do you want to analyse (max = 3200)?\n> "))

    launch_process(twitter_screen_name, reference_url, similarity_threshold, max_tweets)
    

    




    
    

