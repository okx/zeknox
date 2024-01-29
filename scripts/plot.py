import matplotlib.pyplot as plt

if __name__ == "__main__":
    log_n_sizes = list(range(13,20))
    # y1 = [181.223,365.356, 582.569, 1015.192, 1842.546, 3418.811 ]  ## cpu
    # y2 = [208.512, 214.915, 220.078, 293.711, 252.278, 371.174]  ## gpu

    y1 = [0.164, 0.335, 0.705, 1.474, 3.129, 7.17, 15.585 ]
    y2 = [0.038, 0.055, 0.082, 0.136, 0.242, 0.387, 0.682]
   
    y_delta = [(b - a)/b * 100 for a, b in zip(y1, y2)]
    plt.plot(log_n_sizes, y1, label='cpu')
    # plt.plot(log_n_sizes, y2, label='original plonky2 NTT')
    plt.plot(log_n_sizes, y2, label='gpu')
    # plt.plot(log_n_sizes, y_delta, label='improvement factor')
    plt.legend()
    plt.xlabel('log_n_size')
    plt.ylabel('ms')
    plt.show()