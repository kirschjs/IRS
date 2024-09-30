import os
import shutil
import subprocess
import sys
import time

import numpy as np

###### thje following need tp be set to apropriate va;ues
dbg = False  # turn debugging on
csf = True  # am I runninmg on csf?
print("########################")
print("dbg (debug)  is ", dbg)
print("csf (on CSF) is ", csf)
print("########################")

NEWLINE_SIZE_IN_BYTES = -1
numeric_format = 'float64'

MPIRUNcmd = shutil.which("mpirun")

if MPIRUNcmd == None:
    MPIRUNcmd = '/usr/lib64/mpich/bin/mpirun'
    print('<which mpirun> did not return a path.\n Assuming default: %s' %
          MPIRUNcmd)
else:
    print('Using mpirun command : ', MPIRUNcmd)
if os.path.isfile(MPIRUNcmd) == False:
    print(
        'No mpi executable not found at %s.\n hHnt: locate mpirun and change in <settings.py> finds it!'
        % MPIRUNcmd)
    exit(-1)


def run_external(fileName):
    """Run an external file using subprocess; report and fail on error (py 3.6 for CSF)"""
    try:
        process = subprocess.run(fileName, check=True)
        #return should have fields  stdout and stderr (byte sequence)
        if dbg:
            print("Ran ", fileName, " returncode= ", process.returncode)
            if process.stdout:
                print(" with stdout=", process.stdout.decode("utf-8"))
            if process.stderr:
                print(" with stderr=", process.stderr.decode("utf-8"))
    except subprocess.CalledProcessError as e:
        print("Execution of subprocess failed: ", e, file=sys.stderr)
        exit(-1)
    except:
        print("Execution failed: ", sys.exc_info()[0], file=sys.stderr)
        exit(-1)
    #output = subprocess.run(file,capture_output=True)
    #if (output.returncode != 0):
    #    print('While running ',file)
    #    print('Return code:', output.returncode)
    #    print('Output:',output.stdout.decode("utf-8"))
    #    exit(-1)


def du(path):
    """disk usage in human readable format (e.g. '2,1GB')"""
    if csf:
        tmp = 0
    else:
        tmp = subprocess.check_output(
            'du -s %s 2>&1 | grep -v "Permission denied"' % path,
            shell=True  #'2>&1' | grep -v "Permission denied"
        ).split()[0].decode('utf-8')
    return tmp


def freeSpace(path):
    """ free space in bytes, i.e. 400000000000 = 400GB"""
    total, used, free = shutil.disk_usage(path)
    return total, used, free


def cartesian_coord(*arrays):
    grid = np.meshgrid(*arrays)
    coord_list = [entry.ravel() for entry in grid]
    points = np.vstack(coord_list).T
    return points
    #a = np.arange(2)  # fake data
    #print(cartesian_coord(*3 * [a]))
    #1 run <PSI_parallel.py> for boundsatekanal und streukas
    #2 1st <A3_lit_par.py>   run
    #3 2nd <A3_lit_par.py>   run


def testDiskUsage(path, minFree):
    try:
        total, used, free = shutil.disk_usage(path)
        cnt = 0
        while free < minFree and cnt < 5:
            cnt = cnt + 1
            print(
                '%s holds %d bytes (%s percent full). Waiting for 10s to shrink.'
                % path, total, int(free / total * 100))
            time.sleep(10)
            total, used, free = shutil.disk_usage(path)
    except:
        print("(warning) disk-usage-assessment failure.")
        exit(-1)
