# Dan's Computer Vision Experiments

_"Just a bunch of experiments and tests on face recognition, computer vision and such..."_

For more details, see these articles: 

* [Face Clustering with Python](http://logicalgenetics.com/face-clustering-with-python/)
* [Face Clustering with Neural Networks and K-Means](http://logicalgenetics.com/face-clustering-with-neural-networks-and-k-means/)

The order to run the files is something like:

1. `process_face_images.py`:  Specify the input glob.  This script will detect all the faces in the photos matching the glob and place them in a temporary folder called 'faces' to be used by subsequent steps
2. `build_database.py`:  This will build a JSON data file containing the face filenames and corresponding feature vectors.  You can specify whether to use old school 'brute force' feature detection or a neural network
3. `choose-k.py`:  This will show you the cost function for K-means and give you a clue as to the best value for k
4. `cluster.py`:  Does the actual face clustering using K means.  You need to specify K.
5. Fire up a local web server (`python -m http.server 8000` works well) in the `gui` folder and check out the results.  The table view is better than the rather clunky force directed graph!

Nothing in this repo is really attributable to me - I just glued together the work of much wiser human beings.

![](http://logicalgenetics.com/wp-content/uploads/2018/09/Screenshot-2018-09-12-08.07.40.png)
