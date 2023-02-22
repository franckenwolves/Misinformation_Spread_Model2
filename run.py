from virus_on_network.server import server
import sys
import socketserver
import socket


for i in range(8521, 8524):
    try:
        server.launch(i)
        port = i
    except:
        print(i, " is occupied")
"""
s = socket.socket()
server.launch()
port = 8521
"""
### START OF CODE FOR BATCH RUNNER

from virus_on_network.model import *
from virus_on_network.server import *
from numpy import arange
#import matplotlib.pyplot as plt
from mesa.batchrunner import BatchRunner


'''
fixed_params = {
    "virus_check_frequency": 0.4,
    "avg_node_degree": 3,
    "initial_outbreak_size": 1,
    "exposed_chance": 0.3,
    #recovery_chance=0.3,
    "gain_skeptical_chance": 0.5,
    "skeptical_level": 0.2,
}
variable_params = {
    "num_nodes": range(90, 100, 5),
    "virus_spread_chance": arange(0.0, 1.1, .5)
}
num_iterations = 5
num_steps = 1
batch_run = BatchRunner(VirusOnNetwork,
                        fixed_parameters=fixed_params,
                        variable_parameters=variable_params,
                        iterations=num_iterations,
                        max_steps=num_steps,
                        model_reporters=
                        {
                            #some form of csv file writer perhaps?...
                        }
)
batch_run.run_all()'''
#pid = process.pid
#kill(pid, signal.SIGKILL)
#s.close()
