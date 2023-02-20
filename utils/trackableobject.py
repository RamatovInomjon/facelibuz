class TrackableObject:
	def __init__(self, objectID, centroid, head, startY):
		self.objectID = objectID
		self.centroids = [centroid]
		self.head = head
		self.counted = False
		self.startY = startY