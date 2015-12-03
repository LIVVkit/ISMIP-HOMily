#!/usr/bin/env python2

"""
This script gathers all the ISMIP-HOM experiments' data.
"""

import os
import numpy
import scipy
import fnmatch

# Location of data
#TODO: argparse this.
ismip_data = './ismip_all'

# defaults for cism grid
ewn = 40    
nsn = 40

#--------------------------
# ISMIP-HOM data constants 
#--------------------------
#NOTE: Not including 'aas1' as the files do not conform to the data standards.
full_stokes = ['aas2','cma1','fpa2','ghg1','jvj1','mmr1','oga1','rhi1',
               'rhi3','spr1','ssu1','yko1']

lmla = ['ahu1','ahu2','bds1','cma2','fpa1','fsa1','mbr1','rhi2','tpa1']
l1l2 = ['dpo1','rhi4']
l1l1 = ['lpe1','rhi5']
ltsml = ['mtk1']
higher_order = lmla + l1l2 + l1l1 + ltsml

sia = ['oso1']

header = {'a':['x_hat','y_hat','vx_surf','vy_surf','tau_xz','tau_yz','del_p'],
          'b':['x_hat','vx_surf','vz_surf','tau_xz','del_p'],
          'c':['x_hat','y_hat','vx_surf','vy_surf','vz_surf','vx_base',
               'vy_base','tau_xz','tau_yz','del_p'],
          'd':['x_hat','vx_surf','vz_surf','vx_base','tau_xz','del_p'],
          'e':['x_hat','vx_surf','vz_surf','tau_xz','del_p'],
          'f':['x_hat','y_hat','z_surf','vx','vy','vz']
         }
sizes = {'a': [5, 10, 20, 40, 80, 160],
         'b': [5, 10, 20, 40, 80, 160],
         'c': [5, 10, 20, 40, 80, 160],
         'd': [5, 10, 20, 40, 80, 160],
         'e': [],
         'f': [100],
        }


class ismip_datum:
    """A class to hold each model's data"""
    def __init__(self, data_file):
        self.df = data_file
        self.M, self.E, self.L = self.parse_file(data_file)
        
        # classify
        if self.M in full_stokes:
            self.order = 'full_stokes'
        elif self.M in higher_order:
            self.order = 'higher_order'
        elif self.M in sia:
            self.order = 'sia'
        else:
            self.order = 'unknown'
        
        # load the data
        if self.order != 'unknown':
            self.array = self.load_data()
        else:
            self.array = []


    def parse_file(self, data_file):
        """
        Parse the ismip_hom data filenames. They should look like: NNNMELLL.tex, where NNN = model name, 
             M = model number, E = experiment, LLL = three numbers denoting: 
                 experiments a-d: the length of L in km. 
                 experiment e: 000 for non-sliding and 001 for the experiment with 
                               the zone of zero basal raction
                 experiment f: the slip ratio; either 000 or 001.
        """
        code_name = os.path.basename(os.path.splitext(data_file)[0])
        return (code_name[0:4], code_name[4], code_name[5:]) # (model, experiment, length)

    def load_data(self):
        data = numpy.loadtxt(self.df)
        return data

    def display(self):
        print("Data file: "+self.df)
        print("Order: "+self.order)
        print("Model: "+self.M)
        print("Experiment: "+self.E)
        print("Length: "+self.L)


#----------------------
# CISM ISMIP-HOM grids 
#----------------------
class cism_grids:
    """A class to create a CISM grid for comparison."""
    def __init__(self, size):
        self.size = size
        self.nx = ewn
        self.ny = nsn
        self.dx = float(self.size)*1000./float(self.nx)
        self.dy = float(self.size)*1000./float(self.ny)
        
        self.x = [(i+0.5)*self.dx for i in range(self.nx)] # unstaggered grid
        self.y = [(j+0.5)*self.dy for j in range(self.ny)]
        self.y_grid, self.x_grid = scipy.meshgrid(self.y[:], self.x[:], indexing='ij')
        
        self.x_stag = [(i+1)*self.dx for i in range(self.nx-1)] # staggered grid 
        self.y_stag = [(j+1)*self.dy for j in range(self.ny-1)]
        self.y_stag_grid, self.x_stag_grid = scipy.meshgrid(self.y_stag[:], self.x_stag[:], indexing='ij')

        self.x_hat = [i/(self.size*1000) for i in self.x]
        self.y_hat = [i/(self.size*1000) for i in self.y]
        self.y_hat_grid, self.x_hat_grid = scipy.meshgrid(self.y_hat[:], self.x_hat[:], indexing='ij')
        
        self.x_hat_stag = [i/(self.size*1000) for i in self.x]
        self.y_hat_stag = [i/(self.size*1000) for i in self.y]
        self.y_hat_stag_grid, self.x_hat_stag_grid = scipy.meshgrid(self.y_hat_stag[:], self.x_hat_stag[:], indexing='ij')

#-----------
# The files 
#-----------
def recursive_glob(tree, pattern):
    matches = []
    for base, dirs, files in os.walk(tree):
        goodfiles = fnmatch.filter(files, pattern)
        matches.extend(os.path.join(base, f) for f in goodfiles)
    return matches

data_files = recursive_glob(ismip_data, '*.txt')

all_data = []
for i, df in enumerate(data_files):
    all_data.append(ismip_datum(df))
    #all_data[i].display()
    #print("------")
    
grd = []
for sz in sizes['a']:
    grd.append(cism_grids(sz))

grids = {'a': grd,
         'b': grd,
         'c': grd,
         'd': grd,
         'e': [],
         'f': [cism_grids(sizes['f'][0])],
        }


