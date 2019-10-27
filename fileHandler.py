import re
import sys


folder = sys.argv[1]
fileToRead = sys.argv[2]

writingFile = open(folder+"/"+"new"+fileToRead, "w+")
readingfile = open(folder+"/"+fileToRead, "r+")

line = readingfile.readline()

infoList = []
number = 99

while(line):
	infoList = infoList + line.split(' ')
	line = readingfile.readline()


finalList = []
n = 2

for item in infoList:
	if(item[0] == 'l'):
		finalList.append(item)
	elif(len(item) > 12):
		finalList.append(item[:-n + 1])
		finalList.append(item[-n:])
		
		if(number == int(item[-n:])):
			print(number, item[-n:])
			n = n + 1
			if number == 99:
				number = 999
			else:
				number = 9999
	else:
		finalList.append(item)

lines = []

print(len(infoList))
while len(finalList)!=0:
	try:
		lines.append(finalList[0] + " " + finalList[1] + " " + finalList[2] + "\n")
		del finalList[:3]
	except Exception as e:
		break

for line in lines:
	writingFile.write(line)

writingFile.close()