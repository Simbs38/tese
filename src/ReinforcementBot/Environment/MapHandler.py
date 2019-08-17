import json

'''
Considerations about the map structure:
	MapLevel registers every step of the level, and it keeps on updating every time the player explores the level
	MapLevel was a StartPos witch indicates the coord (0,0) in the game, since the game tiles work in a XoY axis sistem in witch you can explore to the negative and positive side of each axis
	MapLevel registers the player position in order to use it later in the GetState method
	MapLevel was a currentPos in order to know the position we need to change in the map matrix after reading it from the json file

	Whenever the map growns to the negative side, the startposition should increment in order to keep note of the coord(0,0)
	Since the game considers:
		y ^ -> lines
		x > -> collumns
	The struct should consider that, and the y coords will be the lines of the struct(the first coord), and the x the collumns(the second coord)
'''



class MapLevel:
	def __init__(self):
		self.map = [['.']]
		self.StartPos = (0,0)
		self.currentPos = (0,0)
		self.PlayerPos = (0,0)

	def AddLinesBefore(self, n):
		size = len(self.map[0])
		for i in range(n):
			line = []
			for j in range(size):
				line.append('.')
			self.map.insert(0,line)
			self.StartPos = (self.StartPos[0]+1, self.StartPos[1])
			self.PlayerPos = (self.PlayerPos[0]+1, self.PlayerPos[1])
		
	def AddLinesAfter(self, n):
		size = len(self.map[0])
		for i in range(n):
			line = []
			for j in range(size):
				line.append('.')
			self.map.append(line)
		
	def AddColumnBefore(self, n):
		size = len(self.map)
		for i in range(n):
			for j in range(size):
				self.map[j].insert(0, '.')
			self.StartPos  = (self.StartPos[0], self.StartPos[1] + 1)
			self.PlayerPos = (self.PlayerPos[0], self.PlayerPos[1]+1)


	def AddColumnAfter(self, n):
		size = len(self.map)
		for i in range(n):
			for j in range(size):
				self.map[j].append('.')
		
	def AddCell(self,cell):
		self.UpdatePosition(cell)
		self.FormatMap(cell)
		self.ParseCell(cell)
		
	def UpdatePosition(self, cell):
		if 'x' in cell:
			self.currentPos = (cell['y'], cell['x'])
		else:
			self.currentPos = (self.currentPos[0], self.currentPos[1] + 1)
		
	def FormatMap(self, cell):
		if self.currentPos[0] < 0:
			LinesToAddBefore =  - (self.currentPos[0] + self.StartPos[0])
			if LinesToAddBefore > 0:
				self.AddLinesBefore(LinesToAddBefore)
		elif self.currentPos[0] > 0:
			LinesToAddAfter = (self.StartPos[0] + self.currentPos[0]) - len(self.map) + 1
			if LinesToAddAfter  > 0:
				self.AddLinesAfter(LinesToAddAfter)

		if self.currentPos[1] < 0:
			ColumnsToAddBefore = -(self.currentPos[1] + self.StartPos[1])
			if ColumnsToAddBefore > 0:
				self.AddColumnBefore(ColumnsToAddBefore)
		elif self.currentPos[1] > 0:
			ColumnsToAddAfter = (self.StartPos[1] + self.currentPos[1]) - len(self.map[0]) + 1
			if ColumnsToAddAfter > 0:
				self.AddColumnAfter(ColumnsToAddAfter)
		
	def ParseCell(self, cell):
		if 'g' in cell:
			self.map[self.currentPos[0] + self.StartPos[0]][self.currentPos[1] + self.StartPos[1]] = cell['g']
			if cell['g'] == '@':
				self.PlayerPos = (self.currentPos[0] + self.StartPos[0], self.currentPos[1] + self.StartPos[1])
		
	def GetState(self, lines, colls):
		if lines > self.PlayerPos[0]:
			self.AddLinesBefore(lines - self.PlayerPos[0])
		if lines > (len(self.map) -  (self.PlayerPos[0])):
			self.AddLinesAfter(lines - (len(self.map) - self.PlayerPos[0]))
		if colls > self.PlayerPos[1]:
			self.AddColumnBefore(colls - self.PlayerPos[1])
		if colls > (len(self.map[0]) - (self.PlayerPos[1])):
			self.AddColumnAfter(colls - (len(self.map[0]) - self.PlayerPos[1]))

		MapState = []

		for i in range(lines * 2):
			line = self.map[self.PlayerPos[0] + i - lines]
			line = line[self.StartPos[1] - colls: self.StartPos[1] + colls]
			MapState.append(line)

		return MapState

class MapHandler:
	def __init__(self):
		self.map = [MapLevel()]
		self.currentLevel = 0

	def LowerOneLevel(self, msg = None):
		self.currentLevel += 1
		if(len(self.map)<(self.currentLevel+1)):
			self.Map.append(MapLevel())
		
		if msg!=None:
			self.UpdateMap(msg)

	def UpOneLevel(self, msg = None):
		self.currentLevel -= 1
		self.UpdateMap(msg)

	def UpdateMap(self, msg):
		cells = msg['cells']
		for cell in cells:
			self.map[self.currentLevel].AddCell(cell)
			
	def GetState(self, lines, colls):
		return self.map[self.currentLevel].GetState(lines, colls)