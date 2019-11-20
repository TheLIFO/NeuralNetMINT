
arr=[4 ,2, 4, 1, 5, 3, 16, 6, 17, 19, 4, 13, 5, 3, 10, 10, 13, 6, 2, 1, 5, 15, 13, 19, 16, 9, 13, 1, 7, 18, 20, 13, 9, 7, 2, 10, 8, 18, 4, 7, 5, 8, 10, 13, 7, 18, 19, 2, 19, 8, 10, 10, 17, 6, 6, 20, 20, 11, 10, 11, 13, 9, 7, 1, 10, 5, 12, 16, 10, 7, 15, 13, 12, 10, 1, 1, 4, 2, 16, 10, 20, 17, 11, 19, 19, 20, 9, 10, 17, 9, 18, 8, 10, 18, 8, 19, 16, 17, 3, 1]
#arr = [2, 5, 3, 4, 3, 2, 5, 5, 3, 4, 2, 2, 2]
arr2D = [[0] * len(arr) for i in range(max(arr))]

for col in range(len(arr)):
    for row in range(arr[col]):
        arr2D[row][col] = 1
w = 0
print('\n'.join([''.join(['{:4}'.format(item) for item in row]) for row in reversed(arr2D)]))
for row in range(max(arr)):
    #go through all rows
    #    
    rf = False
    count = False
    wi = 0
    for col in range(len(arr)):
        if (arr2D[row][col] == 1) and not rf:
            rf = True
        if (arr2D[row][col] == 0) and rf:
            #then count
            count = True
            wi += 1
        if (arr2D[row][col] == 1) and count:
            count = False
            w += wi
            wi = 0
            #stop counting
           
print (w)