from utils.arg_parser import *

def build_condor_instructions(
        filename, username, email, input_dir, output_dir, executable, args_list):
    
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
        
        f.write('Error = $(OutputDir)/err.$(Process)\n')
        f.write('Log = $(OutputDir)/log.$(Process)\n')
        f.write('Output = $(OutputDir)/out.$(Process)\n\n')
        
        f.write('GetEnv = true\n\n')
        
        f.write(f'notify_user = {email}\n')
        f.write('notification = always\n\n')
        
        f.write('# End of the header\n\n')
        
        for i, args in enumerate(args_list, start=1):
            f.write(f'# Condor process : {i}\n')
            f.write(f'Arguments = ${args}\n')
            f.write('Queue 1\n\n')


# Example usage
filename = 'condor_instructions.condor'
username = 'user5'
email = 'nicolas.gonzalez@epfl.ch'
input_dir = '/home/indy-stg3/$(User)/MAproject'
output_dir = '/home/indy-stg3/$(User)/MAproject/logs'
executable = 'condor_exec.sh'

arguments = args()
print_testing(arguments)
dict_args = vars(arguments)


args_list = []

if arguments.test == 'perturbation':
    for g in dict_args['G']:
        for w in dict_args['W']:
            for p in dict_args['P']:
                d = dict_args.copy()
                d['G'] = [g]
                d['W'] = [w]
                d['P'] = [p]
                args_list.append(parse_args_to_string(d))

elif arguments.test == 'gaussian noise':
    for g in dict_args['G']:
        for w in dict_args['W']:
            d = dict_args.copy()
            d['G'] = [g]
            d['W'] = [w]
            d.pop('P')
            args_list.append(parse_args_to_string(d))

elif arguments.test == 'clustering gaussian noise':
    for g in dict_args['G']:
        d = dict_args.copy()
        d['G'] = [g]
        d.pop('W')
        d.pop('P')
        args_list.append(parse_args_to_string(d))

build_condor_instructions(
    filename, username, email, input_dir, output_dir, executable, args_list)
