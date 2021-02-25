import matplotlib.pyplot as plt
from os.path import isfile, exists
from os import mkdir

OUTPUT_FOLDER = "output"

def filename_nmbr():
    if not exists(OUTPUT_FOLDER):
        mkdir(OUTPUT_FOLDER)
    i = 0
    filename = OUTPUT_FOLDER + "/cost-0.pdf"
    while isfile(filename):
        i += 1
        filename = OUTPUT_FOLDER + "/cost-{}.pdf".format(i)
    return i
    

def plot_cost(costs, validations=[], filename="cost.pdf"):
    try:
        plt.figure()
        plt.plot(costs, "-o", c="b", label="Training cost")
        if validations[0]:
            plt.plot(validations, "-o", c="r", label="Validation cost")
            plt.legend()
        plt.grid(True)
        plt.savefig(OUTPUT_FOLDER + "/" + filename)
    finally:
        plt.close()
        
def plot_params(params, filename="params.pdf"):
    try:
        plt.figure()
        plt.plot(params, "-")
        plt.grid(True)
        plt.savefig(OUTPUT_FOLDER + "/" + filename)
    finally:
        plt.close()