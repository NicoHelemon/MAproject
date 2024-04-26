import argparse

def str_to_bool(v):
    if v in ['True', 'False', '']:
        return v == 'True'
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def build_condor(
        test, args_list):
    
    with open(f'utils/condor/{test}.condor', 'w') as f:        
        f.write('User = user5\n')
        f.write('Universe = vanilla\n\n')
        
        f.write(f'InputDir = /home/indy-stg3/$(User)/MAproject\n')
        f.write(f'OutputDir = /home/indy-stg3/$(User)/MAproject\n\n')
        
        f.write(f'Executable = $(InputDir)/utils/condor/{test}.sh\n')
        f.write('InitialDir = $(InputDir)\n\n')
        
        f.write(f'Error = $(OutputDir)/logs/{test}/err.$(Process)\n')
        f.write(f'Log = $(OutputDir)/logs/{test}/log.$(Process)\n')
        f.write(f'Output = $(OutputDir)/logs/{test}/out.$(Process)\n\n')
        
        f.write('GetEnv = true\n\n')
        
        f.write('# End of the header\n\n')
        
        for i, (py_args, tip) in enumerate(args_list, start=1):
            tip += ['net.dat', 'deg.dat', 'comm.dat', 'cs.dat']
            f.write(f'# Condor process : {i}\n')
            transfer_input_files = ', '.join([f'$(InputDir)/{file}' for file in tip])
            f.write(f'transfer_input_files = {transfer_input_files}\n')
            f.write(f'transfer_output_files = results\n')
            f.write(f'Arguments = {py_args}\n')
            f.write('Queue 1\n\n')


def build_sh(executable, test, n_args):
    with open(f'utils/condor/{executable}', 'w') as f:
        f.write('#!/bin/sh\n')
        args_str = ' '.join([f'${i}' for i in range(1, n_args + 1)])
        f.write(f'export PYTHONPATH=/home/indy-stg3/user5/MAproject\n')
        f.write(f'python3 /home/indy-stg3/user5/MAproject/{test} {args_str}\n')