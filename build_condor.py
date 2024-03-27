from utils.arg_parser import *
from utils.tests import *
from utils.static import P_MAP
from pathlib import Path

def build_condor_instructions(
        filename, username, email, input_dir, output_dir, executable, sub_tests_list):
    
    with open(filename, 'w') as f:
        f.write('####################\n')
        f.write('##\n')
        f.write('## .condor file\n')
        f.write('##\n')
        f.write('####################\n\n')
        
        f.write('# Your username is used in pathname\n')
        f.write(f'User = {username}\n\n')
        
        f.write('Universe = vanilla\n\n')
        
        f.write(f'InputDir = {input_dir}\n')
        f.write(f'OutputDir = {output_dir}\n\n')
        
        f.write(f'Executable = $(InputDir)/{executable}\n')
        f.write('InitialDir = $(InputDir)\n\n')
        
        f.write('Error = $(OutputDir)/logs/err.$(Process)\n')
        f.write('Log = $(OutputDir)/logs/log.$(Process)\n')
        f.write('Output = $(OutputDir)/logs/out.$(Process)\n\n')
        
        f.write('GetEnv = true\n\n')

        f.write('transfer_input_files = $(InputDir)/net.dat\n\n')
        f.write('when_to_transfer_output = ON_EXIT\n')
        
        f.write(f'notify_user = {email}\n')
        f.write('notification = always\n\n')
        
        f.write('# End of the header\n\n')
        
        for i, (args, out_path) in enumerate(sub_tests_list, start=1):
            f.write(f'# Condor process : {i}\n')
            f.write(f'transfer_output_files = {out_path}\n')
            f.write(f'Arguments = {args}\n')
            f.write('Queue 1\n\n')


def build_condor_sh(filename, n_args):
    with open(filename, 'w') as f:
        f.write('#!/bin/sh\n')
        f.write('# conda init\n')
        f.write('# conda deactivate\n')
        f.write('# conda init\n')
        f.write('# conda activate myenv\n')
        args_str = ' '.join([f'${i}' for i in range(1, n_args + 1)])
        f.write(f'python3 /home/indy-stg3/user5/MAproject/testing.py {args_str}\n')

instructions = 'condor_instructions.condor'
username = 'user5'
email = 'nicolas.gonzalez@epfl.ch'
input_dir = '/home/indy-stg3/$(User)/MAproject'
output_dir = '/home/indy-stg3/$(User)/MAproject'
executable = 'condor_exec.sh'

args = args()
print("About to condor-parallelize the following test:")
print(test_description_str(args))
dict_args = vars(args)

args_list = []
out_path_list = []

if args.test == 'perturbation':
    for g in dict_args['G']:
        for w in dict_args['W']:
            for p in dict_args['P']:
                d = dict_args.copy()
                d['G'] = [g]
                d['W'] = [w]
                d['P'] = [p]
                p = P_MAP[p].name
                out_path_list.append(Perturbation().out_path(g, w, p))
                args_list.append(parsed_args_to_string(d))

elif args.test == 'gaussian noise':
    for g in dict_args['G']:
        for w in dict_args['W']:
            d = dict_args.copy()
            d['G'] = [g]
            d['W'] = [w]
            d.pop('P')
            out_path_list.append(GaussianNoise().out_path(g, w))
            args_list.append(parsed_args_to_string(d))

elif args.test == 'clustering gaussian noise':
    for g in dict_args['G']:
        d = dict_args.copy()
        d['G'] = [g]
        d.pop('W')
        d.pop('P')
        out_path_list.append(ClusteringGaussianNoise().out_path(g))
        args_list.append(parsed_args_to_string(d))

for out_path in out_path_list:
    Path(out_path).mkdir(parents=True, exist_ok=True)

sub_tests_list = zip(args_list, out_path_list)

build_condor_instructions(
    instructions, username, email, input_dir, output_dir, executable, sub_tests_list)

build_condor_sh(
    executable, len(args_list[0].split()))