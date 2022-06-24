# Twitter Face Ratio Computer

## Scope

This project is a data visualization project  aiming to compute the ratio of appearance of a given face among all the tweets of a given account.

## Workflow

1. Get the *n* latest tweets of a given Twitter account by using the official Twitter API
2. Download all the images and videos shared in the tweets
3. Extract all the faces from the images and videos by using [OpenCV cascade classifier](https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html)
    - For the videos we focus on 1 pane out of 100 in order to lighten the process
    - We apply a [DBScan](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html) clustering on the faces popping out and we keep one face per cluster
4. Encode the faces based on the [face_recognition](https://pypi.org/project/face-recognition/) library
5. Sort the faces based on their similarity with the encoding of a reference image
6. Grey out the faces whose similarity is below a threshold of 0.9 (customisable)
7. Build a mosaic of the transformed faces

## Usage

- Create a Python virtual environment (> 3.9) and activate it
- Create a [Dev API account (V1.1)](https://developer.twitter.com/en/docs/twitter-api/v1) on Twitter and fill the `credentials.json` accordingly
- `pip install -r requirements.txt`
- `python main.py`

### Example

```
python main.py

Please indicate a Twitter screen name to analyse:
> BarackObama

Please indicate an URL to the reference face:
> https://pbs.twimg.com/profile_images/1329647526807543809/2SGvnHYV_400x400.jpg

Please insert a similarity threshold (suggested: 0.9):
> 0.9

How many tweets do you want to analyse (max = 3200)?
> 200

Getting the latest 200 tweets from the user...
200 tweets found

Getting images and videos URLS...
68 images and 31 videos found

Extracting faces from images...
177 faces found on images

Extracting faces from videos...
78 faces found on videos

Computing ratios and generating mosaic...

-------------------------------

The reference face appears 37 times out of 99 media shared by BarackObama (Ratio: 37.37%)
The reference face appears 37 times out of 204 faces popping up in the media shared by BarackObama (Ratio: 14.51%)

The mosaic has been generated : examples/BarackObama.png
```


![alt text](./examples/BarackObama.png "Example")

## Author

Louis de Viron - [DataText SRL](https://www.datatext.eu)

## Credentials

This tool is mainly based on the following python libraries:
- [opencv-python](https://pypi.org/project/opencv-python/)
- [face_recognition](https://pypi.org/project/face-recognition/)
