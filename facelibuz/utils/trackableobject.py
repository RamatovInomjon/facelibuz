class TrackableObject:
	def __init__(self, objectID, centroid, path_length=0):
		# store the object ID, then initialize a list of centroids
		# using the current centroid
		self.objectID = objectID
		self.centroids = [centroid]
		self.path_length=path_length
		self.startY = centroid[1]
		self.name = "unknown"
		self.recognized = False
		self.embeddings = None
		self.bbox = None
		self.kps = None
		self.age = None
		self.gender = None
		self.lost_count = 0
		self.live = False

		# initialize a boolean used to indicate if the object has
		# already been counted or not
		self.counted = False