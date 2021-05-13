#Couple skeleton functions to get started with random num generator to start with
import numpy as np
import time

def random_array_generator(array_size, low=0, high=100):
    '''
    random_array_generator - returns numpy random array of defined size over a defined range
    size of array defined by array_size, minimum value defined by low (defaults to 0) and max value defined by high (defaults to 100)
    '''
    rng = np.random.default_rng()
    return rng.integers(low, high, size=array_size)


def timer_function(sort_function, array_size):
    '''
    timer_fincution - assists benchmarking for different sorting algorithms, returns average over 10 trials of sorting algorithm
    sorting function to test defined by argument sort_function
    size of array to test sorting function with defined by array_size

    array is randomly generated using random_array_generator function
    calls to sorting function being tested are enclosed by time.time() calls to measure time spent on the sorting algorithm
    elapsed time is measured 10 times and the average roudned to 3 decimal places is returned
    '''
    time_elapsed_list = [] #list to track time elapsed over 10 runs
    for i in range(10):
        test_array = random_array_generator(array_size,0,1000) #generate random numbers over range of 0 to 1000
        print(test_array)
        start_time = time.time()*1000 #start time before function called, multiplied by 1000 to convert to microseconds
        test = sort_function(test_array)
        end_time = time.time()*1000 #end time after function finished, multiplied by 1000 to convert to microseconds
        print(test_array)
        time_elapsed = end_time - start_time #total time elapsed
        time_elapsed_list.append(time_elapsed)
    avg_time_elapsed = sum(time_elapsed_list)/len(time_elapsed_list) #get the average time elapsed
    return round(avg_time_elapsed,3) #round to 3 decimal places

#based off bubble sort from realpython article - https://realpython.com/sorting-algorithms-python/
def bubble_sort(array):
    '''
    bubble_sort - performs buble sort on an array in place
    takes single argument for array to be sorted
    '''
    #store the length of the array, this will be used to control the loops
    arr_len = len(array)

    #outer for loop loops the length of the array - 1 times, the -1 saves us an unecessary loop as if
    #we did range(arr_len) it would be already sorted when it does the final loop, -1 removes this issue
    for i in range(arr_len-1):

        #finished_sort is reset to true on every iteration of the outer loop
        #if finished_sort is True after exiting inner loop the outer loop is broken out of and function returns
        finished_sort = True

        #as iterator i increases the amount of values to be checked in the inner loop decreases
        #due to the values at the end of the array being sorted and no longer need to be checked
        for j in range(arr_len-i-1):
            #if current index j is greater than next index item swap
            if array[j] > array[j+1]:
                array[j], array[j+1] = array[j+1], array[j] #swap values at index j and j+1
                #have performed a sort during this loop so set finished_sort flag to false
                finished_sort = False

        #exit outer for loop and return array
        if finished_sort:
            break
    return array

#based off quick sort from - https://www.askpython.com/python/examples/quicksort-algorithm
#for quicksort we will use two functions, one for handling the pivot, second for performing the sort
def pivot(input_array, start, end):
    '''
    pivot - auxiliary function used by quick_sort function, sorts array into two halves lower and higher than pivot value and returns the index where pivot point is
    the array to be pivoted is defined by input_array
    the starting position for the range to pivot over defined by start
    the end position for the range to pivot over defined by end
    it performs it's operations on the array in-place so don't need to return the array but does return the index of where the pivot now sits
    '''
    #pivot on first element in the array
    pivot = input_array[start]
    #low position is 1+pivot indexin several
    low = start + 1
    #high is last inex in array
    high = end

    while True:
        #checks low is less than high and that high is greater than the pivot
        while low <= high and input_array[high] >= pivot:
            #moves high backwards through array as we have value at index high greater than the pivot
            high = high - 1
        #checks low is less than high and that low is less than the pivot
        while low <= high and input_array[low] <= pivot:
            #moves low forwards through array
            low = low + 1
        #low will only go greater than high once it's sorted all values lower than the pivot value
        if low <= high:
            #swap values at index low and high with one another
            input_array[low], input_array[high] = input_array[high], input_array[low]
        #low has become greater than high therefore array has been sorted into low and high halves so exit the loop
        else:
            break
    #lastly swap the pivot element in the array with value for high, this is last index that is in low half
    #from the while loop we don't break out of it till low exceeds high
    input_array[start], input_array[high] = input_array[high], input_array[start]
    return high

#based off quick sort from - https://www.askpython.com/python/examples/quicksort-algorithm
def quick_sort(input_array, start=0, end=None):
    '''
    quick_sort - main function for performing quick sort algorithm, takes in array to sort and sorts it in place, returns nothing
    the array to sort is defined by input_array
    it takes a start position to sort over defined by start (defaults to 0)
    an end position to sort over defined by end (defaults to None)
    this function has optional arguments as it is recursive, on first call to quick_sort function these arguments don't need to be supplied
    on recursive calls these arguments will be provided as it will be performing quick_sort on either half off the pivot
    '''
    #for using timer_function have applied default arguments for start and end
    #if end is none set it to 1 minus the length of the input_array, otherwise have applied end in recursive call
    if end == None:
        end = len(input_array)-1
    if start >= end:
        return True

    #call pivot
    piv = pivot(input_array, start, end)

    #recursively sort left half of array
    quick_sort(input_array, start, piv-1)

    #recursively sort right half of array
    quick_sort(input_array, piv+1, end)

#based off count sort from - https://www.mygreatlearning.com/blog/counting-sort/#t10
def count_sort(input_array, max_value = None):
    '''
    count_sort - this function performs a counting sort on an input array not in-place
    the array to be sorted is defined by input_array
    the maximum value is an optional argument defined by max_value (defaults to None)

    if optional argument not provided it will work this out itself as for counting sort to work it needs to know the max value in the array
    '''
    #if no max_value argument supplied get it using the list function max()
    if max_value is None:
        max_value = max(input_array)

    #some code inspiration from here - https://www.w3resource.com/python-exercises/data-structures-and-algorithms/python-search-and-sorting-exercise-10.php
    #create a list containing only 0's, each 0 represents a value in the range of the input_array
    # using just 0 in an empty list multiply by the maximum value in the array to create a list of 0's
    count_list = [0] * (max_value + 1)
    sorted_list = []

    #loop through the input_array, using the current item from input_array as the index of count_list,
    # increment by 1
    for item in input_array:
        count_list[item] += 1

    #using the enumerate function loop through count_list
    #index keeps track of current index of the list/array
    #item holds the value at that index
    for index, item in enumerate(count_list):
        if item > 0:
            sorted_list.append([index]*item)
    return sorted_list

#based off insert sort from - https://realpython.com/sorting-algorithms-python/#the-insertion-sort-algorithm-in-python
#left and right are necessary arguments due to this function also being used by timsort
def insertion_sort(array, left=0, right=None):
    '''
    insertion_sort - performs insertion sort to sort the input array, it works by its own but also as part of a timsort sorting algorithm
    the input array to sort is defined by array
    position in the array to start sorting at defined by optional argument left (default 0)
    end position in the array to sort over defined by optional argument right (default None)
    '''
    #if no right argument supplied determine it from length of array - 1
    if right is None:
        right = len(array) - 1

    #loop through the array on indexes from the range of left to right
    #as we loop through we try to move the value at index i backwards to sort the array, if it has smaller values before it
    #it stays in place for that loop
    for i in range(left + 1, right + 1):
        #key_item is the current value we want to sort
        key_item = array[i]

        #j always starts with one index behind the key_item,
        #we will be checking key_item against every index before it e.g. trying to move it back through the array by comparing it 
        #against values at index j
        j = i - 1

        #loop through items to the left of key_item, til we find an index that has smaller value than key_item
        #as we loop through we move the value at index j up and for next iteration j is decrimented
        #this in effect makes room for key_item to replace a value as we shift the value at index j up through the array
        while j >= left and array[j] > key_item:
            #set the value at index j+1 = to value at index j
            #this is done to shift the values up through the array
            array[j + 1] = array[j]
            #step back through the array
            j -= 1

        #having looped backwards through the array from the current key_item
        #and come across the first value smaller than key_item
        #set key_item as the value at the index j+1 as everything at j and below is sorted correctly compared to key_item
        array[j + 1] = key_item

    #lastly return the array, this algorithm is in-place so we don't need to assign a variable to capture the output of this function
    return array

def merge(left, right):
    '''
    merge - this function merges two lists/arrays together, returning a sorted merged array
    the first array to be merged is defined by left
    the second array to be merged is defined by right
    this function is used by the hybrid sorting algorithm function timsort as it is a hybrid algorithm, but could also be applied to a merge_sort function
    '''

    #if either left or right is empty notghin to merge so return
    if len(left) == 0:
        return right
    elif len(right) == 0:
        return left
    #result will store the final merged array
    result = []
    index_left = index_right = 0

    #loop through left and right arrays until length of result array is equal to the length of the left and right
    while len(result) < len(left) + len(right):
        #this assumes the left and right arrays are already sorted
        #if current item from left is smaller than right append that to result array
        #otherwise append current item from right
        if left[index_left] <= right[index_right]:
            result.append(left[index_left])
            index_left += 1 #increment index_left
        else:
            result.append(right[index_right])
            index_right += 1 #increment index_right

        #if index_right or index_left equal the length of their array
        #then have finished merging in that array, so can simply add the remaining array to the
        #end of result array
        if index_right == len(right):
            result += left[index_left:]
            #result.append(left[index_left:])
            break
        elif index_left == len(left):
            result += right[index_right:]
            break

    return result

#based off timsort from - https://realpython.com/sorting-algorithms-python/#the-timsort-algorithm-in-python
def timsort(array):
    '''
    timsort - takes in a array to be sorted and sorts it in place
    takes in a single argument for the array to be sorted defined by array

    this is a hybrid sorting algorithm so makes use of first insertion sort on slices of the input array (32 in this case)
    the sorted slices of the array are then merged together using the merge logic from a merge sort algorithm
    '''
    #min_run set to 32 here, this determines the size of slices sorted using insertion sort
    #also determines the size of blocks to later be merge sorted
    min_run = 32
    #n stores the length of the array
    n = len(array)

    #loop from 0 to the size of the array in steps of 32
    #this slices up the input array and allows it to perform
    #insertion sort on slices of the array as insertion sort handles small arrays well
    for i in range(0, n, min_run):
        insertion_sort(array, i, min((i + min_run - 1), n - 1))

    #Having sorted slices of the array, merge the sorted slices
    #size doubles on each iteration until size exceeds length of the array
    #each iteration it will merge and sort each of the slices that were sorte
    #by insertion sort above
    size = min_run
    while size < n:

        #loop through slices of the input array
        #merging slices together as it goes
        for start in range(0, n, size * 2):
            #midpoint and end used to split current selection into
            #left and right arrays
            midpoint = start + size - 1
            #end is the minimmum of next slice step or last index to the array
            #as we don't want to specify an index out of bounds
            end = min((start + size * 2 - 1), (n-1))
            #using defined merge function the two arrays left and right will be merged
            merged_array = merge(
                left=list(array[start:midpoint + 1]),
                right=list(array[midpoint + 1:end + 1]))

            #replace values from array with merged values
            #
            array[start:start + len(merged_array)] = merged_array

        #at the end of each iteration of the while loop double size
        #per iteration it will be merge sorting an increasing slice of the array to be sorted
        # e.g. first iteration merge sort slices [0:32] and [32:64],
        # next loop it will merge sort slices [0:64] and [64:128] and so on
        size *= 2
    return array

if __name__ == '__main__':
    '''
    main - the main function passes each of the sorting algorithms implemented as arguments to the timer function along with different array sizes
    these sorting algorithms are each tested 10 times with different random arrays with the average elapsed time measured in miliseconds
    the output of each being printed using formatting to present in a tabular neat layout the run time of each algorithm across different array sizes
    '''

    bubble_avg_time_elapsed_n_100 = timer_function(bubble_sort, 100)
    '''
    bubble_avg_time_elapsed_n_250 = timer_function(bubble_sort, 250)
    bubble_avg_time_elapsed_n_500 = timer_function(bubble_sort, 500)
    bubble_avg_time_elapsed_n_750 = timer_function(bubble_sort, 750)

    bubble_avg_time_elapsed_n_1000 = timer_function(bubble_sort, 1000)
    bubble_avg_time_elapsed_n_2000 = timer_function(bubble_sort, 2000)
    bubble_avg_time_elapsed_n_4000 = timer_function(bubble_sort, 4000)
    bubble_avg_time_elapsed_n_6000 = timer_function(bubble_sort, 6000)
    bubble_avg_time_elapsed_n_8000 = timer_function(bubble_sort, 8000)
    bubble_avg_time_elapsed_n_10000 = timer_function(bubble_sort, 10000)
    print(f"{'Size':<15} {'100':<12} {'250':<12} {'500':<12} {'750':<12} {'1000':<12} {'2000':<12} {'4000':<12} {'6000':<12} {'8000':<12} {'10000':<12}")
    print(f"{'Bubble Sort':<15} {bubble_avg_time_elapsed_n_100:<12} {bubble_avg_time_elapsed_n_250:<12} {bubble_avg_time_elapsed_n_500:<12} {bubble_avg_time_elapsed_n_750:<12} {bubble_avg_time_elapsed_n_1000:<12} {bubble_avg_time_elapsed_n_2000:<12} {bubble_avg_time_elapsed_n_4000:<12} {bubble_avg_time_elapsed_n_6000:<12} {bubble_avg_time_elapsed_n_8000:<12} {bubble_avg_time_elapsed_n_10000:<12}")
    '''
    '''
    qs_avg_time_elapsed_n_100 = timer_function(quick_sort, 100)
    qs_avg_time_elapsed_n_250 = timer_function(quick_sort, 250)
    qs_avg_time_elapsed_n_500 = timer_function(quick_sort, 500)
    qs_avg_time_elapsed_n_750 = timer_function(quick_sort, 750)
    qs_avg_time_elapsed_n_1000 = timer_function(quick_sort, 1000)
    qs_avg_time_elapsed_n_2000 = timer_function(quick_sort, 2000)
    qs_avg_time_elapsed_n_4000 = timer_function(quick_sort, 4000)
    qs_avg_time_elapsed_n_6000 = timer_function(quick_sort, 6000)
    qs_avg_time_elapsed_n_8000 = timer_function(quick_sort, 8000)
    qs_avg_time_elapsed_n_10000 = timer_function(quick_sort, 10000)
    print(f"{'Quick Sort':<15} {qs_avg_time_elapsed_n_100:<12} {qs_avg_time_elapsed_n_250:<12} {qs_avg_time_elapsed_n_500:<12} {qs_avg_time_elapsed_n_750:<12} {qs_avg_time_elapsed_n_1000:<12} {qs_avg_time_elapsed_n_2000:<12} {qs_avg_time_elapsed_n_4000:<12} {qs_avg_time_elapsed_n_6000:<12} {qs_avg_time_elapsed_n_8000:<12} {qs_avg_time_elapsed_n_10000:<12}")

    cs_avg_time_elapsed_n_100 = timer_function(count_sort, 100)
    cs_avg_time_elapsed_n_250 = timer_function(count_sort, 250)
    cs_avg_time_elapsed_n_500 = timer_function(count_sort, 500)
    cs_avg_time_elapsed_n_750 = timer_function(count_sort, 750)
    cs_avg_time_elapsed_n_1000 = timer_function(count_sort, 1000)
    cs_avg_time_elapsed_n_2000 = timer_function(count_sort, 2000)
    cs_avg_time_elapsed_n_4000 = timer_function(count_sort, 4000)
    cs_avg_time_elapsed_n_6000 = timer_function(count_sort, 6000)
    cs_avg_time_elapsed_n_8000 = timer_function(count_sort, 8000)
    cs_avg_time_elapsed_n_10000 = timer_function(count_sort, 10000)
    print(f"{'Count Sort':<15} {cs_avg_time_elapsed_n_100:<12} {cs_avg_time_elapsed_n_250:<12} {cs_avg_time_elapsed_n_500:<12} {cs_avg_time_elapsed_n_750:<12} {cs_avg_time_elapsed_n_1000:<12} {cs_avg_time_elapsed_n_2000:<12} {cs_avg_time_elapsed_n_4000:<12} {cs_avg_time_elapsed_n_6000:<12} {cs_avg_time_elapsed_n_8000:<12} {cs_avg_time_elapsed_n_10000:<12}")

    is_avg_time_elapsed_n_100 = timer_function(insertion_sort, 100)
    is_avg_time_elapsed_n_250 = timer_function(insertion_sort, 250)
    is_avg_time_elapsed_n_500 = timer_function(insertion_sort, 500)
    is_avg_time_elapsed_n_750 = timer_function(insertion_sort, 750)
    is_avg_time_elapsed_n_1000 = timer_function(insertion_sort, 1000)
    is_avg_time_elapsed_n_2000 = timer_function(insertion_sort, 2000)
    is_avg_time_elapsed_n_4000 = timer_function(insertion_sort, 4000)
    is_avg_time_elapsed_n_6000 = timer_function(insertion_sort, 6000)
    is_avg_time_elapsed_n_8000 = timer_function(insertion_sort, 8000)
    is_avg_time_elapsed_n_10000 = timer_function(insertion_sort, 10000)
    print(f"{'Insertion Sort':<15} {is_avg_time_elapsed_n_100:<12} {is_avg_time_elapsed_n_250:<12} {is_avg_time_elapsed_n_500:<12} {is_avg_time_elapsed_n_750:<12} {is_avg_time_elapsed_n_1000:<12} {is_avg_time_elapsed_n_2000:<12} {is_avg_time_elapsed_n_4000:<12} {is_avg_time_elapsed_n_6000:<12} {is_avg_time_elapsed_n_8000:<12} {is_avg_time_elapsed_n_10000:<12}")

    ts_avg_time_elapsed_n_100 = timer_function(timsort, 100)
    ts_avg_time_elapsed_n_250 = timer_function(timsort, 250)
    ts_avg_time_elapsed_n_500 = timer_function(timsort, 500)
    ts_avg_time_elapsed_n_750 = timer_function(timsort, 750)
    ts_avg_time_elapsed_n_1000 = timer_function(timsort, 1000)
    ts_avg_time_elapsed_n_2000 = timer_function(timsort, 2000)
    ts_avg_time_elapsed_n_4000 = timer_function(timsort, 4000)
    ts_avg_time_elapsed_n_6000 = timer_function(timsort, 6000)
    ts_avg_time_elapsed_n_8000 = timer_function(timsort, 8000)
    ts_avg_time_elapsed_n_10000 = timer_function(timsort, 10000)
    print(f"{'Timsort':<15} {ts_avg_time_elapsed_n_100:<12} {ts_avg_time_elapsed_n_250:<12} {ts_avg_time_elapsed_n_500:<12} {ts_avg_time_elapsed_n_750:<12} {ts_avg_time_elapsed_n_1000:<12} {ts_avg_time_elapsed_n_2000:<12} {ts_avg_time_elapsed_n_4000:<12} {ts_avg_time_elapsed_n_6000:<12} {ts_avg_time_elapsed_n_8000:<12} {ts_avg_time_elapsed_n_10000:<12}")
    '''
    '''
    The following section has been commented out as not part of the project brief and it includes several libraries which aren't necessary as part of the project,
    but it demonstrates how data was collected into dataframes (with pandas) and chart's generated using seaborn libraries lineplot
    '''
    '''
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    #ggplot style for plots
    plt.style.use('ggplot')
    df = pd.DataFrame(columns=["Average Run Time (milliseconds)", "Array Size", "Algorithm"])

    df = df.append({"Average Run Time (milliseconds)":bubble_avg_time_elapsed_n_100, "Array Size":100, "Algorithm":"Bubble Sort"}, ignore_index=True)
    df = df.append({"Average Run Time (milliseconds)":bubble_avg_time_elapsed_n_250, "Array Size":250, "Algorithm":"Bubble Sort"}, ignore_index=True)
    df = df.append({"Average Run Time (milliseconds)":bubble_avg_time_elapsed_n_500, "Array Size":500, "Algorithm":"Bubble Sort"}, ignore_index=True)
    df = df.append({"Average Run Time (milliseconds)":bubble_avg_time_elapsed_n_750, "Array Size":750, "Algorithm":"Bubble Sort"}, ignore_index=True)
    df = df.append({"Average Run Time (milliseconds)":bubble_avg_time_elapsed_n_1000, "Array Size":1000, "Algorithm":"Bubble Sort"}, ignore_index=True)
    df = df.append({"Average Run Time (milliseconds)":bubble_avg_time_elapsed_n_2000, "Array Size":2000, "Algorithm":"Bubble Sort"}, ignore_index=True)
    df = df.append({"Average Run Time (milliseconds)":bubble_avg_time_elapsed_n_4000, "Array Size":4000, "Algorithm":"Bubble Sort"}, ignore_index=True)
    df = df.append({"Average Run Time (milliseconds)":bubble_avg_time_elapsed_n_6000, "Array Size":6000, "Algorithm":"Bubble Sort"}, ignore_index=True)
    df = df.append({"Average Run Time (milliseconds)":bubble_avg_time_elapsed_n_8000, "Array Size":8000, "Algorithm":"Bubble Sort"}, ignore_index=True)
    df = df.append({"Average Run Time (milliseconds)":bubble_avg_time_elapsed_n_10000, "Array Size":10000, "Algorithm":"Bubble Sort"}, ignore_index=True)

    df = df.append({"Average Run Time (milliseconds)":qs_avg_time_elapsed_n_100, "Array Size":100, "Algorithm":"Quick Sort"}, ignore_index=True)
    df = df.append({"Average Run Time (milliseconds)":qs_avg_time_elapsed_n_250, "Array Size":250, "Algorithm":"Quick Sort"}, ignore_index=True)
    df = df.append({"Average Run Time (milliseconds)":qs_avg_time_elapsed_n_500, "Array Size":500, "Algorithm":"Quick Sort"}, ignore_index=True)
    df = df.append({"Average Run Time (milliseconds)":qs_avg_time_elapsed_n_750, "Array Size":750, "Algorithm":"Quick Sort"}, ignore_index=True)
    df = df.append({"Average Run Time (milliseconds)":qs_avg_time_elapsed_n_1000, "Array Size":1000, "Algorithm":"Quick Sort"}, ignore_index=True)
    df = df.append({"Average Run Time (milliseconds)":qs_avg_time_elapsed_n_2000, "Array Size":2000, "Algorithm":"Quick Sort"}, ignore_index=True)
    df = df.append({"Average Run Time (milliseconds)":qs_avg_time_elapsed_n_4000, "Array Size":4000, "Algorithm":"Quick Sort"}, ignore_index=True)
    df = df.append({"Average Run Time (milliseconds)":qs_avg_time_elapsed_n_6000, "Array Size":6000, "Algorithm":"Quick Sort"}, ignore_index=True)
    df = df.append({"Average Run Time (milliseconds)":qs_avg_time_elapsed_n_8000, "Array Size":8000, "Algorithm":"Quick Sort"}, ignore_index=True)
    df = df.append({"Average Run Time (milliseconds)":qs_avg_time_elapsed_n_10000, "Array Size":10000, "Algorithm":"Quick Sort"}, ignore_index=True)

    df = df.append({"Average Run Time (milliseconds)":cs_avg_time_elapsed_n_100, "Array Size":100, "Algorithm":"Count Sort"}, ignore_index=True)
    df = df.append({"Average Run Time (milliseconds)":cs_avg_time_elapsed_n_250, "Array Size":250, "Algorithm":"Count Sort"}, ignore_index=True)
    df = df.append({"Average Run Time (milliseconds)":cs_avg_time_elapsed_n_500, "Array Size":500, "Algorithm":"Count Sort"}, ignore_index=True)
    df = df.append({"Average Run Time (milliseconds)":cs_avg_time_elapsed_n_750, "Array Size":750, "Algorithm":"Count Sort"}, ignore_index=True)
    df = df.append({"Average Run Time (milliseconds)":cs_avg_time_elapsed_n_1000, "Array Size":1000, "Algorithm":"Count Sort"}, ignore_index=True)
    df = df.append({"Average Run Time (milliseconds)":cs_avg_time_elapsed_n_2000, "Array Size":2000, "Algorithm":"Count Sort"}, ignore_index=True)
    df = df.append({"Average Run Time (milliseconds)":cs_avg_time_elapsed_n_4000, "Array Size":4000, "Algorithm":"Count Sort"}, ignore_index=True)
    df = df.append({"Average Run Time (milliseconds)":cs_avg_time_elapsed_n_6000, "Array Size":6000, "Algorithm":"Count Sort"}, ignore_index=True)
    df = df.append({"Average Run Time (milliseconds)":cs_avg_time_elapsed_n_8000, "Array Size":8000, "Algorithm":"Count Sort"}, ignore_index=True)
    df = df.append({"Average Run Time (milliseconds)":cs_avg_time_elapsed_n_10000, "Array Size":10000, "Algorithm":"Count Sort"}, ignore_index=True)

    df = df.append({"Average Run Time (milliseconds)":is_avg_time_elapsed_n_100, "Array Size":100, "Algorithm":"Insertion Sort"}, ignore_index=True)
    df = df.append({"Average Run Time (milliseconds)":is_avg_time_elapsed_n_250, "Array Size":250, "Algorithm":"Insertion Sort"}, ignore_index=True)
    df = df.append({"Average Run Time (milliseconds)":is_avg_time_elapsed_n_500, "Array Size":500, "Algorithm":"Insertion Sort"}, ignore_index=True)
    df = df.append({"Average Run Time (milliseconds)":is_avg_time_elapsed_n_750, "Array Size":750, "Algorithm":"Insertion Sort"}, ignore_index=True)
    df = df.append({"Average Run Time (milliseconds)":is_avg_time_elapsed_n_1000, "Array Size":1000, "Algorithm":"Insertion Sort"}, ignore_index=True)
    df = df.append({"Average Run Time (milliseconds)":is_avg_time_elapsed_n_2000, "Array Size":2000, "Algorithm":"Insertion Sort"}, ignore_index=True)
    df = df.append({"Average Run Time (milliseconds)":is_avg_time_elapsed_n_4000, "Array Size":4000, "Algorithm":"Insertion Sort"}, ignore_index=True)
    df = df.append({"Average Run Time (milliseconds)":is_avg_time_elapsed_n_6000, "Array Size":6000, "Algorithm":"Insertion Sort"}, ignore_index=True)
    df = df.append({"Average Run Time (milliseconds)":is_avg_time_elapsed_n_8000, "Array Size":8000, "Algorithm":"Insertion Sort"}, ignore_index=True)
    df = df.append({"Average Run Time (milliseconds)":is_avg_time_elapsed_n_10000, "Array Size":10000, "Algorithm":"Insertion Sort"}, ignore_index=True)

    df = df.append({"Average Run Time (milliseconds)":ts_avg_time_elapsed_n_100, "Array Size":100, "Algorithm":"Timsort"}, ignore_index=True)
    df = df.append({"Average Run Time (milliseconds)":ts_avg_time_elapsed_n_250, "Array Size":250, "Algorithm":"Timsort"}, ignore_index=True)
    df = df.append({"Average Run Time (milliseconds)":ts_avg_time_elapsed_n_500, "Array Size":500, "Algorithm":"Timsort"}, ignore_index=True)
    df = df.append({"Average Run Time (milliseconds)":ts_avg_time_elapsed_n_750, "Array Size":750, "Algorithm":"Timsort"}, ignore_index=True)
    df = df.append({"Average Run Time (milliseconds)":ts_avg_time_elapsed_n_1000, "Array Size":1000, "Algorithm":"Timsort"}, ignore_index=True)
    df = df.append({"Average Run Time (milliseconds)":ts_avg_time_elapsed_n_2000, "Array Size":2000, "Algorithm":"Timsort"}, ignore_index=True)
    df = df.append({"Average Run Time (milliseconds)":ts_avg_time_elapsed_n_4000, "Array Size":4000, "Algorithm":"Timsort"}, ignore_index=True)
    df = df.append({"Average Run Time (milliseconds)":ts_avg_time_elapsed_n_6000, "Array Size":6000, "Algorithm":"Timsort"}, ignore_index=True)
    df = df.append({"Average Run Time (milliseconds)":ts_avg_time_elapsed_n_8000, "Array Size":8000, "Algorithm":"Timsort"}, ignore_index=True)
    df = df.append({"Average Run Time (milliseconds)":ts_avg_time_elapsed_n_10000, "Array Size":10000, "Algorithm":"Timsort"}, ignore_index=True)

    plt.figure(figsize=(16,6))
    normalplot = sns.lineplot(data=df, x="Array Size", y="Average Run Time (milliseconds)", hue="Algorithm", style="Algorithm", markers=True)
    plt.title('Sort Algorithm Elapsed Time v. Array Size')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()

    plt.figure(figsize=(16,6))
    logplot = sns.lineplot(data=df, x="Array Size", y="Average Run Time (milliseconds)", hue="Algorithm", style="Algorithm", markers=True)
    logplot.set(yscale="log")
    plt.title('Sort Algorithm Elapsed Time v. Array Size (Log)')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()
    '''