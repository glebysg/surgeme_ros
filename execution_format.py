import numpy as np 
import csv 
import math 

data_that_matters = []
with open('Communication Message for Executiion - Taurus_sim.csv') as csvfile:
	readCSV = csv.reader(csvfile)
	for row in readCSV:
		data = np.array([row[3],row[5],row[7],row[10],row[8]])#grab start new peg transfer, prediction , pass_type , target_pole, intial peg poses 
		data_that_matters.append(data)
data_that_matters = np.array(data_that_matters)
# print("grab start new peg transfer, prediction , pass_type , target_pole")
data_that_matters = data_that_matters[1:,:]
# print np.array(data_that_matters)

input_data = []
for x in data_that_matters:
	a = int(x[0])
	# print a
	b = int(x[1])
	if x[2] == 'L-R':
		x[2] = 'left'
	elif x[2] == 'R-L':
		x[2] = 'right'
	d = int(x[3])
	inp = [a,b,d]
	input_data.append(inp)
input_data = np.array(input_data) #start nw tranfer, prediction, peg=dest
# print(input_data)
transfer_ids = np.where(input_data[:,0]==1)
transfer_ids = np.array(transfer_ids)
batches = []
for x in transfer_ids[0,:]:
	# print x
	batch = input_data[x:x+7,1]
	batches.append(batch)
# print batches
#batches has the surgeme order for each transfer_id 
# find tranfer_id of input to get destination and l-r or r-l 

count = 1
counter = np.array([0,0,0,0,0,0,0])
counter.astype(float)
for x in transfer_ids[0,:]:
	print("Input Format: start_tranfer_index, prediction , pass_type , target_pole, order")
	print("Input to be sent: ",data_that_matters[x,:])
	print("Peg poses: ",data_that_matters[x,-1])
	print("Limb: ",data_that_matters[x,2])
	print("Source and Dest: ",input_data[x,2])
	print("Count: ",count)
	sequence = batches[count-1]
	if sequence[0] == 0:
		counter[0] = counter[0]+1
	if sequence[1] == 1:
		counter[1] = counter[1]+1
	if sequence[2] == 2:
		counter[2] = counter[2]+1
	if sequence[3] == 3:
		counter[3] = counter[3]+1
	if sequence[4] == 4:
		counter[4] = counter[4]+1
	if sequence[5] == 5:
		counter[5] = counter[5]+1
	if sequence[6] == 6:
		counter[6] = counter[6]+1
 
	# for surgeme in sequence:
	# 	if surgeme== 0:
	# 		count0 = count0+1
	# 	print("Surgeme: ",surgeme)
	count = count+1
	print("***********************************************************************************************************")
print("***********************************************************************************************************")
print("")
print("***********************************************************************************************************")
print(counter)
print((counter)/float(count-1))
print("***********************************************************************************************************")


################ Please place pegs in this position 
