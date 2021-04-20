#Couple skeleton functions to get started with random num generator to start with
import numpy as np
import time

def random_list_generator(list_size, low=0, high=10):
    rng = np.random.default_rng()
    return rng.integers(low, high, size=list_size)

def timer_function(sort_function, array_size):
    time_elapsed_list = []
    for i in range(10):
        test_array = random_list_generator(array_size,0,100)
        start_time = time.time()*1000
        test = sort_function(test_array)
        end_time = time.time()*1000
        time_elapsed = end_time - start_time
        time_elapsed_list.append(time_elapsed)
    avg_time_elapsed = sum(time_elapsed_list)/len(time_elapsed_list)
    return round(avg_time_elapsed,3)

#some inspiration taken from realpython article - https://realpython.com/sorting-algorithms-python/
def bubble_sort(array):
    
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

#https://www.askpython.com/python/examples/quicksort-algorithm
#for quicksort we will use two functions, one for handling the pivot, second for performing the sort
def pivot(input_array, start, end):
    
    #pivot on first element in the array
    pivot = input_array[start]
    #low position is 1+pivot index
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

def quick_sort(input_array, start=0, end=None):
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

print("100")
bubble_avg_time_elapsed_n_100 = timer_function(bubble_sort, 100)
print("250")
bubble_avg_time_elapsed_n_250 = timer_function(bubble_sort, 250)
print("500")
bubble_avg_time_elapsed_n_500 = timer_function(bubble_sort, 500)
print("750")
bubble_avg_time_elapsed_n_750 = timer_function(bubble_sort, 750)
print("1000")
bubble_avg_time_elapsed_n_1000 = timer_function(bubble_sort, 1000)
print("2000")
bubble_avg_time_elapsed_n_2000 = timer_function(bubble_sort, 2000)
print("4000")
bubble_avg_time_elapsed_n_4000 = timer_function(bubble_sort, 4000)
print("6000")
bubble_avg_time_elapsed_n_6000 = timer_function(bubble_sort, 6000)
print("8000")
bubble_avg_time_elapsed_n_8000 = timer_function(bubble_sort, 8000)
print("10000")
bubble_avg_time_elapsed_n_10000 = timer_function(bubble_sort, 10000)
print(f"{'Size':<15} {'100':<7} {'250':<7} {'500':<7} {'750':<7} {'1000':<7} {'2000':<7} {'4000':<7} {'6000':<7} {'8000':<7} {'10000':<7}")
print(f"{'Bubble Sort':<15} {bubble_avg_time_elapsed_n_100:<7} {bubble_avg_time_elapsed_n_250:<7} {bubble_avg_time_elapsed_n_500:<7} {bubble_avg_time_elapsed_n_750:<7} {bubble_avg_time_elapsed_n_1000:<7} {bubble_avg_time_elapsed_n_2000:<7} {bubble_avg_time_elapsed_n_4000:<7} {bubble_avg_time_elapsed_n_6000:<7} {bubble_avg_time_elapsed_n_8000:<7} {bubble_avg_time_elapsed_n_10000:<7}")

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
print(f"{'Quick Sort':<15} {qs_avg_time_elapsed_n_100:<7} {qs_avg_time_elapsed_n_250:<7} {qs_avg_time_elapsed_n_500:<7} {qs_avg_time_elapsed_n_750:<7} {qs_avg_time_elapsed_n_1000:<7} {qs_avg_time_elapsed_n_2000:<7} {qs_avg_time_elapsed_n_4000:<7} {qs_avg_time_elapsed_n_6000:<7} {qs_avg_time_elapsed_n_8000:<7} {qs_avg_time_elapsed_n_10000:<7}")