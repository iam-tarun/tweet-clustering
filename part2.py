import re
import random
import pandas as pd
# configurations
config = {
 'k' : [5, 10, 15, 20, 25, 30, 35, 40],
 'dataset': 'https://drive.google.com/uc?id=1wAVFFqbd046uDKbSVPfvonA9iWKbq_es',
 'logging': False,
 'logFilePath': 'log.txt',
 'cleanData': True,
 'data': []
}

# function to pre process the tweets data file 
def process_data_file(filePath: str):
 newFileLines = []
 regex = re.compile("\@.+|http:\/\/*|\.\@.+") # regular expression to match http urls and words start with @
 

 # file = open(filePath, 'r', encoding="cp1252")
 file = pd.read_csv(filePath, delimiter="\t", header=None, encoding="cp1252")
 # read all the lines in the dataset
 for line in file[0].tolist():
  words = [word.lower() for word in (line.split("|")[-1]).split(" ") if not regex.match(word)] # remove the words start with @ and http urls and time ids
  newFileLines.append(" ".join(words) + '\n') # add them into the new file

 config['data'] = newFileLines


# K Means Clustering Model
class K_Means_Clustering:
 def __init__(self, k:int, logging= config['logging'], logFilePath= config['logFilePath']):
  self.k = k
  self.X = None
  self.data = None
  self.k_cluster = {}
  self.logging = logging
  self.logFilePath = logFilePath

 # function to calculate the Jaccard Distance
 def distance(self, arr1: [], arr2: []):
  words1 = set(arr1)
  words2 = set(arr2)
  # 1 - |A intersection B|/|A union B|
  return 1 - (len(words1.intersection(words2))/len(words1.union(words2)))
 
 # function to load the data from the pre processed file
 def load_data(self):
  try:
    self.data = config['data']
    self.X = [word[:-1].split(" ") for word in self.data]
    # print(self.X[0])
  except:
    print("error occurred while loading the data.")
  
 # function to initialize the clusters dict for given value of k 
 def initialize_clusters(self):
  for i in range(self.k):
    self.k_cluster[i] = {
      'center':  [],
      'data': []
    }

 # function to get the index of cluster for which given point is closer
 def getMinCenter(self, x: []):
  minI = self.k
  minDistance = 2
  for i in range(self.k):
   dist = self.distance(x, self.k_cluster[i]['center'])
   if dist < minDistance:
    minDistance = dist
    minI = i
  return minI
 
 # function to get the point that is closer to all the remaining points in a given cluster
 def getMinPoint(self, cluster: []):
  minI = len(cluster)
  minDistance = -1
  # go through each point
  for point in range(len(cluster)):
   totalDistance = 0
   for point2 in range(len(cluster)):
    # calculate the total distance of this point from all other points in the cluster
    if point != point2:
     totalDistance += self.distance(cluster[point], cluster[point2])
   # find the point with smallest total Distance
   if minDistance == -1 or minDistance > totalDistance:
    minDistance = totalDistance
    minI = point
  return minI
 
 # function to calculate the Sum of Squared Error
 def SSE(self):
  error = 0
  for i in range(self.k):
   for point in self.k_cluster[i]['data']:
    error += (self.distance(point, self.k_cluster[i]['center']))**2
  return error

 # function to initialize the clusters with random centroids
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
   
  sse = self.SSE() 
  print(self.k, " clusters with SSE = ", sse) 
  if self.logging:
   logFile = open(self.logFilePath, 'a')
   logFile.write("k value: " + str(self.k) + "\n")
   logFile.write("SSE Error: " + str(sse) + "\n")
   logFile.writelines([(str(i) + "th cluster items: " + str(len(self.k_cluster[i]['data'])) + ", ") for i in range(self.k)] + ["\n"])
   logFile.close()
  
if config['cleanData']:
 process_data_file(config['dataset'])

for k in config['k']:
 model = K_Means_Clustering(k)
 model.load_data()
 model.fit()
