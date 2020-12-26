import os
import sys

# Setting the path one folder up
sys_path = sys.path[0].split("/")
sys_path.pop(-1)
sys.path[0] = "/".join(sys_path)

from nn.tests import test

test()