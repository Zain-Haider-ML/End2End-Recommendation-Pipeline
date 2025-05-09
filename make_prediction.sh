#!/bin/bash

USER_ID=1
MOVIE_ID=1
MODEL="meta"

echo "Sending prediction request to localhost:5000/predict"
curl -X GET "http://127.0.0.1:5000/predict?user_id=$USER_ID&movie_id=$MOVIE_ID&model=$MODEL"
