import multiprocessing,time

def function_square(data):
    time.sleep(2)
    print(data)
    return data

if __name__ == '__main__':
    inputs = list(range(100))
    pool = multiprocessing.Pool(processes=4)
    for i in inputs:
        pool.apply_async(function_square,(i,))
    # pool_outputs = pool.map(function_square, inputs)
    pool.close()
    pool.join()
    # print ('Pool    :', pool_outputs)