import re
import random

def process_data_file(filePath: str):
 newFileLines = []
 regex = re.compile("http://bbc.in/*")
 
 try: 
  file = open(filePath, 'r')
  lines = file.readlines()
  for line in lines:
   words = [word.lower() for word in (line.split("|")[-1]).split(" ") if not regex.match(word)]
   newFileLines.append(" ".join(words) + '\n')
  file.close()

  print("hi")
  newFile = open('data.txt', 'w')
  newFile.writelines(newFileLines)
  newFile.close()

 except:
  print("unable to read the file.\n")

# process_data_file('bbchealth.txt')

class K_Means_Clustering:
 def __init__(self, k:int, dataset: str):
  self.k = k
  self.dataset = dataset
  self.X = None
  self.data = None
  self.k_cluster = {}

 def distance(self, arr1: [], arr2: []):
  words1 = set(arr1)
  words2 = set(arr2)
  return 1 - (len(words1.intersection(words2))/len(words1.union(words2)))
 
 def load_data(self):
  try:
    datafile = open(self.dataset)
    self.data = datafile.readlines()
    self.X = [word[:-1].split(" ") for word in self.data]
    # print(self.X[0])
  except:
    print("error occurred while loading the data.")
  
 def initialize_clusters(self):
  for i in range(self.k):
    self.k_cluster[i] = {
      'center':  [],
      'data': []
    }

 def getMinCenter(self, x: []):
  minI = self.k
  minDistance = 2
  for i in range(self.k):
   dist = self.distance(x, self.k_cluster[i]['center'])
   if dist < minDistance:
    minDistance = dist
    minI = i
  return minI
 
 def getMinPoint(self, cluster: []):
  minI = len(cluster)
  minDistance = -1
  for point in range(len(cluster)):
   totalDistance = 0
   for point2 in range(len(cluster)):
    if point != point2:
     totalDistance += self.distance(cluster[point], cluster[point2])
   if minDistance == -1 or minDistance > totalDistance:
    minDistance = totalDistance
    minI = point
  return minI
 
 def SSE(self):
  error = 0
  for i in range(self.k):
   for point in self.k_cluster[i]['data']:
    error += (self.distance(point, self.k_cluster[i]['center']))**2
  return error

 def fit(self):
  #  step 1 - initializing
  self.initialize_clusters()
  ##    generate random indexes as centroids
  rand_set = set()
  i = 0
  while i < self.k:
   randI = random.randint(0, len(self.X))
   if randI not in rand_set:
    rand_set.add(randI)
    self.k_cluster[i]['center'] = self.X[randI]
    i+=1
   else:
    continue
  
  prev_centroids = set(" ".join(self.k_cluster[i]['center']) for i in range(self.k))
  while True:
   # step 2 - divide data based on the centers
   for dataPoint in self.X:
    self.k_cluster[self.getMinCenter(dataPoint)]['data'].append(dataPoint)
   
   # step 3 - assign new centers based on the cluster data
   for i in range(self.k):
    self.k_cluster[i]['center'] = self.k_cluster[i]['data'][self.getMinPoint(self.k_cluster[i]['data'])]
    
  
   # check if the prev centroids and new centroids are similar or not
   new_centroids = set(" ".join(self.k_cluster[i]['center']) for i in range(self.k))
   if prev_centroids == new_centroids:
    break
   else:
    for i in range(self.k):
     self.k_cluster[i]['data'] = []
    prev_centroids = new_centroids 
   
  # print([self.k_cluster[i]['center'] for i in range(self.k)])
  print(self.SSE())


# print(test.distance("the long march", "ides of march"))
model = K_Means_Clustering(250, 'data.txt')
model.load_data()
model.fit()
