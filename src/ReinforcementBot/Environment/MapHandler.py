import json

class MapLevel:
	def __init__(self):
		self.map = [['.']]
		self.StartPos = (0,0)
		self.currentPos = (0,0)
		self.PlayerPos = (0,0)

	def AddLinesBefore(self, n):
		size = len(self.map[0])
		for i in range(n):
			for j in range(size):
				line.append('.')
			self.map.insert(0,line)
			self.StartPos[0] += 1

	def AddLinesAfter(self, n):
		size = len(self.map[0])
		for i in range(n):
			line = []
			for j in range(size):
				line.append('.')
			self.map.insert(0, line)

	def AddColumnBefore(self, n):
		size = len(self.map)
		for i in range(n):
			for j in range(size):
				map[j].insert(0, '.')
			self.StartPos[1] +=1

	def AddColumnAfter(self, n):
		size = len(self.map)
		for i in range(n):
			for j in range(size):
				map[j].append('.')

	def AddCell(self,cell):
		self.UpdatePosition(cell)
		self.FormatMap(cell)
		self.ParseCell(cell)
		
	def UpdatePosition(self, cell):
		if 'x' in cell:
			self.currentPos[0] = cell['x']
			self.currentPos[1] = cell['y']
		else:
			self.currentPos[1] +=1

	def FormatMap(self, cell):
		if self.currentPos[0] < 0:
			LinesToAddBefore =  - (self.currentPos[0] + self.StartPos[0])
			if LinesToAddBefore > 0:
				AddLinesBefore(LinesToAddBefore)
		elif self.currentPos[0] > 0:
			LinesToAddAfter = len(self.map) - (self.StartPos[0] + self.currentPos[0])
			if LinesToAddAfter  > 0:
				AddLinesAfter(LinesToAddAfter)

		if self.currentPos[1] < 0:
			ColumnsToAddBefore = -(self.currentPos[1] + self.StartPos[1])
			if ColumnsToAddBefore > 0:
				AddColumnBefore(ColumnsToAddBefore)
		elif self.currentPos[1] > 0:
			ColumnsToAddAfter = len(self.map[0])  - (self.StartPos[1] + self.currentPos[1])
			if ColumnsToAddAfter > 0:
				AddColumnAfter(ColumnsToAddAfter)
		
	def ParseCell(self, cell):
		if 'g' in cell:
			self.map[self.currentPos[0] + self.StartPos[0]][self.currentPos[1] + self.StartPos[1]] = cell['g']
		elif 'mf' in cell:
			self.map[self.currentPos[0] + self.StartPos[0]][self.currentPos[1] + self.StartPos[1]] = '.'

		if cell['g'] == '@':
			self.PlayerPos = self.currentPos

	def GetState(self, sizeX, sizeY):
		if sizeX > self.PlayerPos[0]:
			AddLinesBefore(sizeX - PlayerPos[0])
		if sizeX > (len(self.map) -  (self.PlayerPos[0] + self.currentPos[0])):
			AddLinesAfter(sizeX - (len(self.map) - (self.PlayerPos[0] + self.currentPos[0])))
		if sizeY > self.PlayerPos[1]:
			AddColumnBefore(sizeY - PlayerPos[1])
		if sizeY > (len(self.map) - (self.PlayerPos[0] + self.currentPos[0])):
			AddColumnAfter(len(self.map) - (self.PlayerPos[0] + self.currentPos[0]))

		MapState = []

		for i in range(sizeX * 2):
			MapState.append(self.map[self.PlayerPos[0] + i - sizeX][(self.PlayerPos[1] - sizeY): (self.PlayerPos[1] + sizeY)])

		return MapState

class MapHandler:
	def __init__(self):
		self.map = []
		self.currentLevel = 0

	def LowerOneLevel(self, msg = None):
		self.currentLevel += 1
		self.Map.append(Map())
		if msg!=None:
			self.UpdateMap(msg)

	def UpOneLevel(self, msg = None):
		self.currentLevel -= 1
		self.UpdateMap(msg)

	def UpdateMap(self, msg):
		map = json.load()
		cells = map['cells']

		for cell in cells:
			map[currentLevel].UpdatePosition(cell)

	def GetState(self, sizeX, sizeY):
		return self.map[self.currentLevel].GetState(sizeX, sizeY)