import sys
import numpy as np
import scipy
import scipy.io.wavfile

"""
Calculate the Euclidean Distance
"""


def CalculateDistance(x, y):
    return np.linalg.norm(x - y)


"""
Assign the points to their nearest centroids then update the centroids location.
"""


def UpdateCentroids(current_centroids, X):
    # points are assigned
    assigned_points = np.zeros(0)
    for x in X:
        dist = []
        for centroid in current_centroids:
            dist.append(CalculateDistance(x, centroid))
        centroid_index = np.argmin(dist)
        assigned_points = np.append(assigned_points, centroid_index)

    # centroids are updated
    new_centroids = np.zeros(current_centroids.shape)
    num_of_points_in_cluster = np.zeros(current_centroids.shape[0])
    for i in range(assigned_points.size):
        new_centroids[int(assigned_points[i])] += X[i]
        num_of_points_in_cluster[int(assigned_points[i])] += 1
    num_of_points_in_cluster = num_of_points_in_cluster.reshape(num_of_points_in_cluster.size, 1)
    for num in num_of_points_in_cluster:  # bug fix for dividing by 0
        if (num == 0):
            num += 1
    new_centroids /= num_of_points_in_cluster
    return new_centroids


"""
Run the KMeans algorithm for the number of the iterations given, or untill the centroids converge.
return the new values of the centroids.
"""


def KMeans(iterations, x, current_centroids):
    file = open("output.txt", "w")
    for i in range(iterations):
        old_centroids = current_centroids
        # assign the points to their centroids and update the centroids.
        current_centroids = np.round(UpdateCentroids(current_centroids, x))
        print(f"[iter {i}]:{','.join([str(i) for i in current_centroids])}\n")
        file.write(f"[iter {i}]:{','.join([str(i) for i in current_centroids])}\n")
        # stop if the centroids converge
        if np.array_equal(current_centroids, old_centroids):
            break
    file.close()
    return current_centroids


def main():
    sample, centroids = sys.argv[1], sys.argv[2]
    fs, y = scipy.io.wavfile.read(sample)
    x = np.array(y.copy())
    centroids = np.loadtxt(centroids)
    new_values = KMeans(30, x, centroids)
    scipy.io.wavfile.write("compressed.wav", fs, np.array(new_values, dtype=np.int16))


if __name__ == "__main__":
    main()
