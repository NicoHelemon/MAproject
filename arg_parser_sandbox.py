from utils.arg_parser import *

args, parser = args()
print_testing(args)
print(parse_args_to_string(args))

dict_args = vars(args)



for arg in arg_list:
    print(arg)
    print()